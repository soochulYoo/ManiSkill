ALGO_NAME = 'BC_ACT_rgbd'

import argparse
import os
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from act.evaluate import evaluate
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import tyro


def decompose_force(force6: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Combines the two 3D fingertip contact forces (left, right) into one net force, decomposed
    into unit direction (3) + magnitude (1). Input (..., 6) = [left_xyz, right_xyz] -> output
    (..., 4) = [dir_x, dir_y, dir_z, magnitude]."""
    net_force = force6[..., 0:3] + force6[..., 3:6]
    magnitude = torch.linalg.norm(net_force, dim=-1, keepdim=True) 
    direction = net_force / (magnitude + eps)
    log_magnitude = torch.log1p(magnitude)
    contact = (magnitude > 0.1).float()
    return torch.cat([direction, log_magnitude, contact], dim=-1)

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill_ACT"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = 'pickcube.trajectory.rgbd.pd_joint_delta_pos.cpu.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""

    # ACT specific arguments
    lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    kl_weight: float = 10
    """weight for the kl loss term"""
    temporal_agg: bool = True
    """if toggled, temporal ensembling will be performed"""

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = True
    include_force: bool = True
    """Whether to include contact force as its own conditioning token in the model (see
    DETRVAE's force_dim), rather than folding it into the flat state vector like other extra
    fields. The two fingertip contact forces are combined into one net force and decomposed into
    direction (3) + magnitude (1). Requires the demo dataset/env to expose `finger_contact_forces`."""
    use_force_magnitude_head: bool = False
    """Whether to add an auxiliary head that predicts the next timestep's force magnitude from the
    current one (self-supervised: needs only the magnitude sequence, no extra labels), trained
    jointly with the main behavior cloning loss. Only meaningful if include_force is True. This is
    the head Test-Time Training adapts online at eval time (see eval_friction_sweep.py)."""
    force_magnitude_loss_weight: float = 1.0
    """Weight for the auxiliary force-magnitude-prediction loss, only used if
    use_force_magnitude_head is True."""
    force_hidden_dim: int = 64
    """Hidden size of the ForceMagnitudeEncoder (the small MLP that encodes force magnitude and
    that Test-Time Training adapts)."""

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, rgb=True, depth=False, state=True, force=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.include_force = force
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # resize the input image to be at least 224x224
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        force = None
        if self.include_force:
            force = decompose_force(observation["extra"].pop("finger_contact_forces"))
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                resized_rgb = self.transforms(
                    cam_data["rgb"].permute(0, 3, 1, 2)
                )  # (1, 3, 224, 224)
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(
                    depth.permute(0, 3, 1, 2)
                )  # (1, 1, 224, 224)
                images_depth.append(resized_depth)

        rgb = torch.stack(images_rgb, dim=1) # (1, num_cams, C, 224, 224), uint8
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1) # (1, num_cams, C, 224, 224), float16

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_force:
            ret["force"] = force
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = rgb
        elif self.include_rgb and self.include_depth:
            ret["rgb"] = rgb
            ret["depth"] = depth
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = depth
        return ret


class SmallDemoDataset_ACTPolicy(Dataset): # Load everything into memory
    def __init__(self, data_path, num_queries, num_traj, include_depth=True, include_force=True, use_force_magnitude_head=False):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from act.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print('Raw trajectory loaded, start to pre-process the observations...')

        self.include_depth = include_depth
        self.include_force = include_force
        self.use_force_magnitude_head = include_force and use_force_magnitude_head
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # pre-trained models from torchvision.models expect input image to be at least 224x224

        # Pre-process the observations, make them align with the obs returned by the FlattenRGBDObservationWrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories['observations']:
            obs_traj_dict = self.process_obs(obs_traj_dict)
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories['observations'] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict.keys())

        # Pre-process the actions
        for i in range(len(trajectories['actions'])):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in range(self.num_traj):
            episode_len = trajectories['actions'][traj_idx].shape[0]
            self.slices += [
                (traj_idx, ts) for ts in range(episode_len)
            ]

        print(f"Length of Dataset: {len(self.slices)}")

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.delta_control = 'delta' in args.control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        # get state at start_ts only
        state = self.trajectories['observations'][traj_idx]['state'][ts]
        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.num_queries-action_len, 1)], dim=0)

        # get force at start_ts, and (if training the magnitude head) the next num_queries steps of
        # [log_magnitude, contact] as the multi-step self-supervised prediction target -- mirrors
        # act_seq itself being a chunk of num_queries future actions, padded the same way past the
        # end of the episode (repeat the last available value). Observations have episode_len+1
        # entries while ts ranges over the episode_len action indices, so ts+1 is always in-bounds
        # even when the full window isn't.
        if self.include_force:
            force = self.trajectories['observations'][traj_idx]['force'][ts]
            if self.use_force_magnitude_head:
                future_force = self.trajectories['observations'][traj_idx]['force'][ts + 1: ts + 1 + self.num_queries]
                future_len = future_force.shape[0]
                if future_len < self.num_queries:
                    pad = future_force[-1:].repeat(self.num_queries - future_len, 1)
                    future_force = torch.cat([future_force, pad], dim=0)

        # normalize state, force (magnitude component only -- direction is already a unit vector
        # and contact is already a clean 0/1 flag, neither needs/wants z-score normalization), and
        # act_seq
        if not self.delta_control:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            if self.include_force:
                force = force.clone()
                force[3:4] = (force[3:4] - self.norm_stats["force_magnitude_mean"][0]) / self.norm_stats["force_magnitude_std"][0]
                if self.use_force_magnitude_head:
                    future_force = future_force.clone()
                    future_force[:, 3:4] = (future_force[:, 3:4] - self.norm_stats["force_magnitude_mean"][0]) / self.norm_stats["force_magnitude_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # get rgb or rgbd data at start_ts and combine with state to form obs
        if self.include_depth:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            depth = self.trajectories['observations'][traj_idx]['depth'][ts]
            obs = dict(state=state, rgb=rgb, depth=depth)
        else:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][ts]
            obs = dict(state=state, rgb=rgb)
        if self.include_force:
            obs['force'] = force

        item = {
            'observations': obs,
            'actions': act_seq,
        }
        if self.use_force_magnitude_head:
            item['next_force_magnitude'] = future_force[:, 3:4] # (num_queries, 1), normalized log-magnitude
            item['next_force_contact'] = future_force[:, 4:5]   # (num_queries, 1), binary mask, unnormalized
        return item

    def __len__(self):
        return len(self.slices)

    def process_obs(self, obs_dict):
        # get rgbd data
        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        force = None
        if self.include_force:
            # pulled out before flattening so it stays its own modality instead of getting
            # folded into the flat "state" vector along with the rest of obs_dict['extra']
            raw_force = torch.from_numpy(obs_dict['extra'].pop('finger_contact_forces')).float() # (ep_len, 6)
            force = decompose_force(raw_force) # (ep_len, 5) = [dir_x, dir_y, dir_z, log_magnitude, contact]
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            rgb = torch.from_numpy(cam_data["rgb"]) # (ep_len, H, W, 3)
            resized_rgb = self.transforms(
                rgb.permute(0, 3, 1, 2)
            )  # (ep_len, 3, 224, 224); pre-trained models from torchvision.models expect input image to be at least 224x224
            images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = torch.Tensor(cam_data["depth"].astype(np.float32) / 1024).to(torch.float16) # (ep_len, H, W, 1)
                resized_depth = self.transforms(
                    depth.permute(0, 3, 1, 2)
                )  # (ep_len, 1, 224, 224); pre-trained models from torchvision.models expect input image to be at least 224x224
                images_depth.append(resized_depth)
        rgb = torch.stack(images_rgb, dim=1) # (ep_len, num_cams, 3, 224, 224) # still uint8
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1) # (ep_len, num_cams, 1, 224, 224) # float16

        # flatten the rest of the data which should just be state data
        obs_dict['extra'] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict['extra'].items()} # dirty fix for data that has one dimension (e.g. is_grasped)
        obs_dict = common.flatten_state_dict(obs_dict, use_torch=True)

        processed_obs = dict(state=obs_dict, rgb=rgb, depth=depth) if self.include_depth else dict(state=obs_dict, rgb=rgb)
        if self.include_force:
            processed_obs['force'] = force

        return processed_obs

    def get_norm_stats(self):
        all_state_data = []
        all_action_data = []
        all_force_data = [] if self.include_force else None
        for traj_idx, ts in self.slices:
            state = self.trajectories['observations'][traj_idx]['state'][ts]
            act_seq = self.trajectories['actions'][traj_idx][ts:ts+self.num_queries]
            action_len = act_seq.shape[0]
            if action_len < self.num_queries:
                target_pos = act_seq[-1]
                act_seq = torch.cat([act_seq, target_pos.repeat(self.num_queries-action_len, 1)], dim=0)
            all_state_data.append(state)
            all_action_data.append(act_seq)
            if self.include_force:
                all_force_data.append(self.trajectories['observations'][traj_idx]['force'][ts])

        all_state_data = torch.stack(all_state_data)
        all_action_data = torch.concatenate(all_action_data)

        # normalize obs (state) data
        state_mean = all_state_data.mean(dim=0, keepdim=True)
        state_std = all_state_data.std(dim=0, keepdim=True)
        state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        stats = {"action_mean": action_mean, "action_std": action_std,
                 "state_mean": state_mean, "state_std": state_std,
                 "example_state": state}

        if self.include_force:
            # only the (log-)magnitude component gets normalized: direction is already a unit
            # vector and contact is already a clean 0/1 flag, so z-scoring either would just
            # distort a representation that's already well-scaled.
            all_force_data = torch.stack(all_force_data) # (N, 5) = [dir(3), log_magnitude(1), contact(1)]
            force_magnitude_mean = all_force_data[:, 3:4].mean(dim=0, keepdim=True)
            force_magnitude_std = all_force_data[:, 3:4].std(dim=0, keepdim=True)
            force_magnitude_std = torch.clip(force_magnitude_std, 1e-2, np.inf) # clipping
            stats["force_magnitude_mean"] = force_magnitude_mean
            stats["force_magnitude_std"] = force_magnitude_std

        return stats


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space['state'].shape) == 1 # (obs_dim,)
        assert len(env.single_observation_space['rgb'].shape) == 4 # (num_cams, C, H, W)
        assert len(env.single_action_space.shape) == 1 # (act_dim,)
        #assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()

        self.state_dim = env.single_observation_space['state'].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.include_force = args.include_force
        self.use_force_magnitude_head = args.include_force and args.use_force_magnitude_head
        self.force_magnitude_loss_weight = args.force_magnitude_loss_weight
        if self.include_force:
            assert len(env.single_observation_space['force'].shape) == 1 # (force_dim,) = direction(3) + magnitude(1)
            self.force_dim = env.single_observation_space['force'].shape[0]
        else:
            self.force_dim = None
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # CNN backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
            force_dim=self.force_dim,
            force_hidden_dim=args.force_hidden_dim,
            use_force_magnitude_head=self.use_force_magnitude_head,
        )

    def compute_loss(self, obs, action_seq, next_force_magnitude=None, next_force_contact=None):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if args.include_depth:
            obs['depth'] = obs['depth'].float()

        # force data
        if self.include_force:
            obs['force'] = obs['force'].float()

        # forward pass
        a_hat, (mu, logvar), force_magnitude_pred = self.model(obs, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        # store all loss
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        if self.use_force_magnitude_head:
            # self-supervised auxiliary task: predict the next num_queries steps of force
            # magnitude from the current one (same chunk length as the action head). Masked by
            # the *future* contact flag so the loss only trains on windows that actually involve
            # contact -- long no-contact stretches would otherwise dominate the auxiliary loss
            # with the trivial "predict zero" case and drown out the few informative
            # contact-transition regions. Same masked task used for Test-Time Training at eval
            # time (see eval_friction_sweep.py).
            all_mse = F.mse_loss(force_magnitude_pred, next_force_magnitude.float(), reduction='none')
            contact_mask = next_force_contact.float()
            force_magnitude_loss = (all_mse * contact_mask).sum() / contact_mask.sum().clamp(min=1)
            loss_dict['force_magnitude_pred'] = force_magnitude_loss
            loss_dict['loss'] = loss_dict['loss'] + force_magnitude_loss * self.force_magnitude_loss_weight
        return loss_dict

    def get_action(self, obs):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # depth data
        if args.include_depth:
            obs['depth'] = obs['depth'].float()

        # force data
        if self.include_force:
            obs['force'] = obs['force'].float()

        # forward pass
        a_hat, (_, _), _ = self.model(obs) # no action, sample from prior

        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'norm_stats': dataset.norm_stats,
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgbd" if args.include_depth else "rgb", render_mode="rgb_array")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth, force=args.include_force)]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers)

    # dataloader setup
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, args.num_queries, num_traj=args.num_demos, include_depth=args.include_depth, include_force=args.include_force, use_force_magnitude_head=args.use_force_magnitude_head)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = dataset.num_traj

    obs_mode = "rgb+depth" if args.include_depth else "rgb"

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT",
            tags=["act"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # agent setup
    agent = Agent(envs, args).to(device)

    # optimizer setup
    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)

    # LR drop by a factor of 10 after lr_drop iters
    lr_drop = int((2/3)*args.total_iters)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    # Evaluation
    #eval_kwargs = dict(
    #    stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
    #    max_timesteps=gym_utils.find_max_episode_steps_value(envs), device=device, sim_backend=args.sim_backend
    #)
    eval_kwargs = dict(
        stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=args.max_episode_steps, device=device, sim_backend=args.sim_backend
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for cur_iter, data_batch in enumerate(train_dataloader):
        last_tick = time.time()
        # copy data from cpu to gpu
        obs_batch_dict = data_batch['observations']
        obs_batch_dict = {k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()}
        act_batch = data_batch['actions'].cuda(non_blocking=True)
        next_force_magnitude_batch = data_batch['next_force_magnitude'].cuda(non_blocking=True) if args.use_force_magnitude_head else None
        next_force_contact_batch = data_batch['next_force_contact'].cuda(non_blocking=True) if args.use_force_magnitude_head else None

        # forward and compute loss
        loss_dict = agent.compute_loss(
            obs=obs_batch_dict, # obs_batch_dict['state'] is (B, obs_dim)
            action_seq=act_batch, # (B, num_queries, act_dim)
            next_force_magnitude=next_force_magnitude_batch,
            next_force_contact=next_force_contact_batch,
        )
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        timings["update"] += time.time() - last_tick

        # Evaluation
        if cur_iter % args.eval_freq == 0:
            last_tick = time.time()

            ema.copy_to(ema_agent.parameters())

            eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], cur_iter)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')

        if cur_iter % args.log_freq == 0:
            print(f"Iteration {cur_iter}, loss: {total_loss.item()}")
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("charts/backbone_learning_rate", optimizer.param_groups[1]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            if args.use_force_magnitude_head:
                writer.add_scalar("losses/force_magnitude_pred", loss_dict['force_magnitude_pred'].item(), cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)

        # Checkpoint
        if args.save_freq is not None and cur_iter % args.save_freq == 0:
            save_ckpt(run_name, str(cur_iter))

    envs.close()
    writer.close()
