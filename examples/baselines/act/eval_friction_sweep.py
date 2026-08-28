"""
Evaluates and compares trained ACT checkpoints across a sweep of table friction / object density
conditions, tracking success rate and return per condition, and saves a comparison plot.

Compares up to 4 cases (only the ones with a checkpoint path given are run):
  - no_force:          ACT trained without any force input (--ckpt-no-force)
  - force:              ACT trained with force (direction+magnitude) but no auxiliary head (--ckpt-force)
  - force_head_no_ttt:  ACT trained with force + the force-magnitude prediction head, evaluated
                        normally (frozen weights) (--ckpt-force-head)
  - force_head_ttt:     the SAME checkpoint as force_head_no_ttt, but with Test-Time Training:
                        online gradient updates to ONLY the ForceMagnitudeEncoder + prediction
                        head during rollout, using the self-supervised "predict next magnitude
                        from current magnitude" task (the true next magnitude becomes available
                        one env.step() later, no extra labels needed)

Architecture hyperparameters (hidden_dim, nheads, etc.) must match what each checkpoint was
trained with -- they are not saved in the checkpoint itself (matches train_rgbd.py's save_ckpt).
"""
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tyro

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from act.make_env import make_eval_envs
from mani_skill.utils import common, gym_utils
import train_rgbd
from train_rgbd import Agent, FlattenRGBDObservationWrapper

# same sweep as scripts/data_generation/motionplanning_friction_density.sh
DEFAULT_CONDITIONS: List[Tuple[float, float, float]] = [
    (1000.0, 0.5, 0.35),
    (1000.0, 1.0, 0.7),
    (1000.0, 2.0, 1.4),
    (1000.0, 3.3, 2.3),
    (1000.0, 4.5, 3.15),
    (1000.0, 6.0, 4.2),
    (1000.0, 8.0, 5.6),
]

CASE_NAMES = {
    "no_force": "ACT w/o force",
    "force": "ACT w/ force, w/o head",
    "force_head_no_ttt": "ACT w/ force+head, w/o TTT",
    "force_head_ttt": "ACT w/ force+head, w/ TTT",
}


@dataclass
class Args:
    env_id: str = "PushCube-v1"
    control_mode: str = "pd_joint_pos"

    ckpt_no_force: Optional[str] = None
    """checkpoint for the ACT-without-force case. Omit to skip this case."""
    ckpt_force: Optional[str] = None
    """checkpoint for the ACT-with-force-no-head case. Omit to skip this case."""
    ckpt_force_head: Optional[str] = None
    """checkpoint for the ACT-with-force-and-magnitude-head case. Used for BOTH the
    force_head_no_ttt and force_head_ttt cases (same weights, different eval-time behavior).
    Omit to skip both cases."""

    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    """NOTE: must be the literal string "physx_cpu" (not "cpu") for act/make_env.py's CPU
    multiprocessing branch to trigger; passing "cpu" silently falls through to the GPU-oriented
    branch and errors on num_eval_envs > 1 (this is a pre-existing quirk in make_env.py, also
    present in train_rgbd.py's identically-defaulted Args.sim_backend)."""
    seed: int = 1

    # architecture hyperparameters -- must match training. See train_rgbd.py's Args for docs.
    backbone: str = "resnet18"
    position_embedding: str = "sine"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = True
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    dim_feedforward: int = 512
    enc_layers: int = 2
    dec_layers: int = 4
    pre_norm: bool = False
    num_queries: int = 30
    temporal_agg: bool = True
    force_hidden_dim: int = 64
    max_episode_steps: Optional[int] = None
    kl_weight: float = 10
    """unused at eval time (no BC loss computed) but still read by Agent.__init__; keep matching training."""

    ttt_lr: float = 1e-3
    """learning rate for the online Test-Time Training updates to ForceMagnitudeEncoder + the
    magnitude prediction head (force_head_ttt case only)."""

    output_dir: str = "runs/friction_sweep_eval"


def build_agent(env, args: Args, include_force: bool, use_force_magnitude_head: bool, device) -> Agent:
    agent_args = SimpleNamespace(
        **{f.name: getattr(args, f.name) for f in args.__dataclass_fields__.values()},
        include_force=include_force,
        use_force_magnitude_head=use_force_magnitude_head,
        force_magnitude_loss_weight=1.0,
    )
    return Agent(env, agent_args).to(device)


def make_condition_envs(args: Args, condition, include_force, run_name_suffix):
    density, static_f, dynamic_f = condition
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="normalized_dense",
        obs_mode="rgbd" if args.include_depth else "rgb",
        render_mode="rgb_array",
        obj_density=density,
        static_friction=static_f,
        dynamic_friction=dynamic_f,
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth, force=include_force)]
    return make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, None,
                           video_dir=None, wrappers=wrappers)


def evaluate_with_ttt(n, agent: Agent, eval_envs, norm_stats, args: Args, device, use_ttt: bool):
    """Adapted from act/evaluate.py's evaluate(), with an added online Test-Time Training hook.
    When use_ttt is True, at every rollout step we take one gradient step on ONLY
    agent.model.force_magnitude_encoder + agent.model.force_magnitude_pred_head, using the
    self-supervised "predict magnitude[t] from magnitude[t-1]" loss (the true magnitude[t] just
    became available from the env; magnitude[t-1]'s prediction was made last step). The main
    policy (action_head, backbone, transformer, etc.) is never updated during eval."""
    stats = norm_stats
    delta_control = not stats
    assert not delta_control, "friction sweep eval assumes an absolute (non-delta) control mode"

    def _make_force_pre_process(use_numpy: bool):
        # only the (log-)magnitude component (index 3) gets normalized -- direction (0:3) is
        # already a unit vector and contact (4) is already a clean 0/1 flag, matching how the
        # dataset normalizes force at training time.
        mag_mean = stats['force_magnitude_mean'].cpu().numpy() if use_numpy else stats['force_magnitude_mean']
        mag_std = stats['force_magnitude_std'].cpu().numpy() if use_numpy else stats['force_magnitude_std']

        def _process(f_obs):
            f_obs = f_obs.copy() if use_numpy else f_obs.clone()
            f_obs[..., 3:4] = (f_obs[..., 3:4] - mag_mean) / mag_std
            return f_obs
        return _process

    if args.sim_backend == "physx_cpu" or args.sim_backend == "cpu":
        pre_process = lambda s_obs: (s_obs - stats['state_mean'].cpu().numpy()) / stats['state_std'].cpu().numpy()
        force_pre_process = _make_force_pre_process(use_numpy=True) if 'force_magnitude_mean' in stats else None
    else:
        pre_process = lambda s_obs: (s_obs - stats['state_mean']) / stats['state_std']
        force_pre_process = _make_force_pre_process(use_numpy=False) if 'force_magnitude_mean' in stats else None
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    max_timesteps = args.max_episode_steps or gym_utils.find_max_episode_steps_value(eval_envs)
    num_queries = args.num_queries
    action_dim = eval_envs.action_space.shape[-1]
    num_envs = eval_envs.num_envs
    temporal_agg = args.temporal_agg
    if temporal_agg:
        query_frequency = 1
        all_time_actions = torch.zeros([num_envs, max_timesteps, max_timesteps + num_queries, action_dim], device=device)
    else:
        query_frequency = num_queries
        actions_to_take = torch.zeros([num_envs, num_queries, action_dim], device=device)

    ttt_optimizer = None
    if use_ttt:
        # Explicitly freeze everything (vision backbone, transformer, action head, proprio/state
        # projections, ...) and only unfreeze ForceMagnitudeEncoder + the magnitude prediction
        # head, rather than relying only on the optimizer's param group to scope the update (that
        # alone is correct too, since the TTT loss's forward pass never touches the rest of the
        # model, but this makes "everything else stays frozen" an explicit, structural property
        # of the model rather than an implicit consequence of which tensors happen to be in the
        # loss's computation graph).
        for p in agent.parameters():
            p.requires_grad_(False)
        ttt_params = list(agent.model.force_magnitude_encoder.parameters())
        if hasattr(agent.model, "force_magnitude_pred_head"):
            ttt_params += list(agent.model.force_magnitude_pred_head.parameters())
        for p in ttt_params:
            p.requires_grad_(True)
        ttt_optimizer = torch.optim.Adam(ttt_params, lr=args.ttt_lr)
        # Delayed-buffer TTT: force_magnitude_pred_head predicts num_queries steps ahead, so a
        # prediction made at step t can only be checked against ground truth as those future real
        # steps actually happen. We store the raw (log_magnitude, contact) *inputs* at each past
        # step rather than the predictions themselves, and recompute predictions fresh (through
        # the current, possibly already-adapted-this-step weights) whenever we need to validate
        # one against a newly observed true value -- storing the predictions directly and
        # backpropagating through them later would require retaining their autograd graphs for up
        # to num_queries steps each, which is both unsupported for the in-place buffer writes this
        # would need and would otherwise leak memory across a long rollout. Recomputation is cheap
        # since the encoder+head are tiny.
        magnitude_input_history = torch.zeros([num_envs, max_timesteps, 2], device=device)
    else:
        # force_head_no_ttt / force / no_force cases: no gradients at all, frozen inference only.
        for p in agent.parameters():
            p.requires_grad_(False)

    agent.eval()
    eval_metrics = defaultdict(list)
    obs, info = eval_envs.reset(seed=args.seed)
    ts, eps_count = 0, 0
    while eps_count < n:
        # pre-process obs (normalize); this MUST happen before the TTT step below since the
        # magnitude prediction task operates in the same normalized space as training
        obs['state'] = pre_process(obs['state'])
        if force_pre_process is not None and 'force' in obs:
            obs['force'] = force_pre_process(obs['force'])
        obs = {k: common.to_tensor(v, device) for k, v in obs.items()}

        if use_ttt and 'force' in obs:
            cur_mag_contact = obs['force'][:, 3:5].float()  # [log_magnitude, contact], just observed (true) values
            cur_magnitude = cur_mag_contact[:, 0:1]
            cur_contact = cur_mag_contact[:, 1:2]

            start = max(0, ts - num_queries)
            if start < ts:
                # every past input at t' in [start, ts) predicted magnitude[t'+1 .. t'+num_queries];
                # the element targeting THIS step ts is at offset (ts - t' - 1). Recompute all of
                # them fresh (batched together) through the current weights.
                past_inputs = magnitude_input_history[:, start:ts]  # (num_envs, K, 2)
                K = past_inputs.shape[1]
                h = agent.model.force_magnitude_encoder(past_inputs.reshape(num_envs * K, 2))
                preds = agent.model.force_magnitude_pred_head(h).reshape(num_envs, K, num_queries)
                target_offsets = (ts - torch.arange(start, ts, device=device) - 1)  # (K,), in [0, num_queries)
                preds_for_curr_step = torch.gather(
                    preds, dim=2, index=target_offsets.view(1, K, 1).expand(num_envs, K, 1)
                ).squeeze(-1)  # (num_envs, K)

                target = cur_magnitude.expand(-1, K)
                mask = cur_contact.expand(-1, K)
                if mask.sum() > 0:
                    all_mse = F.mse_loss(preds_for_curr_step, target, reduction="none")
                    ttt_loss = (all_mse * mask).sum() / mask.sum().clamp(min=1)
                    ttt_optimizer.zero_grad()
                    ttt_loss.backward()
                    ttt_optimizer.step()

            magnitude_input_history[:, ts] = cur_mag_contact.detach()

        with torch.no_grad():
            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)  # (num_envs, num_queries, action_dim)

            if temporal_agg:
                all_time_actions[:, ts, ts:ts + num_queries] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                actions_populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                actions_populated[max(0, ts + 1 - num_queries):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, actions_populated]
                k = 0.01
                exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step[0]), device=device))
                exp_weights = (exp_weights / exp_weights.sum())
                exp_weights = torch.tile(exp_weights, (num_envs, 1)).unsqueeze(-1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            action = post_process(raw_action)
            if args.sim_backend in ("physx_cpu", "cpu"):
                action = action.cpu().numpy()

        obs, rew, terminated, truncated, info = eval_envs.step(action)
        ts += 1

        if truncated.any():
            assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation"
            # see the matching comment in act/evaluate.py: GPU backend always wraps terminal info
            # in `final_info`, gymnasium>=1.0's own vector envs (CPU backend here) do not.
            if "final_info" in info:
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(common.to_numpy(v))
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
            else:
                # CPU backend: gymnasium>=1.0's default autoreset mode returns the terminal
                # (pre-reset) obs here and only resets on the NEXT step() call -- relying on that
                # would silently consume one extra "phantom" step per episode that throws off
                # ts/all_time_actions indexing on the following episode. Reset explicitly.
                for k, v in info["episode"].items():
                    eval_metrics[k].append(common.to_numpy(v))
                obs, info = eval_envs.reset()
            eps_count += num_envs
            ts = 0
            if temporal_agg:
                all_time_actions = torch.zeros([num_envs, max_timesteps, max_timesteps + num_queries, action_dim], device=device)
            if use_ttt:
                magnitude_input_history = torch.zeros([num_envs, max_timesteps, 2], device=device)

    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics


def run_case(case: str, ckpt_path: str, args: Args, device):
    print(f"\n=== Evaluating case: {CASE_NAMES[case]} ({ckpt_path}) ===")
    include_force = case != "no_force"
    use_force_magnitude_head = case in ("force_head_no_ttt", "force_head_ttt")
    use_ttt = case == "force_head_ttt"

    condition_results = {}
    for condition in DEFAULT_CONDITIONS:
        density, static_f, dynamic_f = condition
        label = f"static={static_f}"
        print(f"  condition density={density} static={static_f} dynamic={dynamic_f}")
        envs = make_condition_envs(args, condition, include_force, run_name_suffix=f"{case}_{static_f}")

        agent = build_agent(envs, args, include_force, use_force_magnitude_head, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["ema_agent"])
        norm_stats = ckpt["norm_stats"]

        metrics = evaluate_with_ttt(args.num_eval_episodes, agent, envs, norm_stats, args, device, use_ttt=use_ttt)
        envs.close()

        success_key = "success_at_end" if "success_at_end" in metrics else ("success_once" if "success_once" in metrics else None)
        success_rate = float(np.mean(metrics[success_key])) if success_key else float("nan")
        return_key = "return" if "return" in metrics else ("r" if "r" in metrics else None)
        mean_return = float(np.mean(metrics[return_key])) if return_key else float("nan")
        print(f"    success_rate={success_rate:.4f} mean_return={mean_return:.4f}")
        condition_results[label] = dict(
            density=density, static_friction=static_f, dynamic_friction=dynamic_f,
            success_rate=success_rate, mean_return=mean_return,
        )
    return condition_results


def plot_results(all_results: dict, output_dir: str):
    conditions = list(next(iter(all_results.values())).keys())
    x = [all_results[list(all_results.keys())[0]][c]["static_friction"] for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for case, condition_results in all_results.items():
        success = [condition_results[c]["success_rate"] for c in conditions]
        ret = [condition_results[c]["mean_return"] for c in conditions]
        axes[0].plot(x, success, marker="o", label=CASE_NAMES[case])
        axes[1].plot(x, ret, marker="o", label=CASE_NAMES[case])

    axes[0].set_xlabel("static friction")
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Success rate vs. table friction")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("static friction")
    axes[1].set_ylabel("mean return")
    axes[1].set_title("Return vs. table friction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "friction_sweep_comparison.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\nSaved plot to {plot_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs(args.output_dir, exist_ok=True)
    # Agent.get_action/compute_loss reference a module-level `args` global in train_rgbd.py (only
    # set there inside its own __main__ block), rather than an instance attribute -- so importing
    # Agent elsewhere requires setting this manually before calling into it.
    train_rgbd.args = args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cases_to_run = []
    if args.ckpt_no_force is not None:
        cases_to_run.append(("no_force", args.ckpt_no_force))
    if args.ckpt_force is not None:
        cases_to_run.append(("force", args.ckpt_force))
    if args.ckpt_force_head is not None:
        cases_to_run.append(("force_head_no_ttt", args.ckpt_force_head))
        cases_to_run.append(("force_head_ttt", args.ckpt_force_head))

    if len(cases_to_run) == 0:
        raise ValueError("No checkpoints given. Pass at least one of --ckpt-no-force/--ckpt-force/--ckpt-force-head.")

    all_results = {}
    for case, ckpt_path in cases_to_run:
        all_results[case] = run_case(case, ckpt_path, args, device)

    results_path = os.path.join(args.output_dir, "friction_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    plot_results(all_results, args.output_dir)
