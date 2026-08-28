# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from act.detr.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ForceComponentEncoder(nn.Module):
    """Small MLP encoder for one force sub-component (direction or magnitude+contact), each kept
    on its own encoder/token rather than being concatenated and projected together. For the
    magnitude encoder specifically: this is the module Test-Time Training adapts. Its output both
    (a) feeds the policy's force-magnitude token and (b) is read by `force_magnitude_pred_head` to
    predict *future* magnitudes, a purely self-supervised task (needs only the magnitude/contact
    sequence itself, no action/success labels) usable both as an auxiliary training loss and, at
    test time, as an online adaptation signal since the true future magnitudes become available a
    few steps later."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries, force_dim=None,
                 force_hidden_dim=64, use_force_magnitude_head=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.backbones = None

        # optional force modality: net contact force decomposed as direction (3) + log-magnitude
        # (1) + a binary contact flag (1), i.e. obs['force'] = [dir_x, dir_y, dir_z, log_magnitude,
        # contact] (see decompose_force in train_rgbd.py). Direction and magnitude get their own
        # encoder and their own decoder token each (rather than being concatenated into one), so
        # e.g. Test-Time Training can adapt the magnitude pathway alone without disturbing the
        # direction pathway. The contact flag rides along with magnitude (it's a direct property
        # of it) into the magnitude encoder, not treated as its own token. Conditions the decoder
        # the same way proprio state does. Only meaningful with a vision backbone (force is not
        # wired into the state-only path used by train.py). Not fed into the CVAE
        # encoder/recognition network below (unlike state) since that network only affects the KL
        # latent used at training time; force needs to influence the decoder's actual action
        # prediction, which is what determines test-time reactive behavior.
        #
        # The magnitude+contact component is routed through its own ForceComponentEncoder rather
        # than projected directly, so that its hidden representation can also be read by an
        # auxiliary future-magnitude prediction head (self-supervised: needs only the
        # magnitude/contact sequence, no extra labels). This shared representation is what Test-Time
        # Training adapts at eval time -- adapting it changes both the auxiliary prediction AND the
        # policy's actual magnitude token, since they both read from it.
        self.force_dim = force_dim
        self.use_force_magnitude_head = use_force_magnitude_head
        if self.force_dim is not None:
            # obs['force'] layout: direction (3) + [log_magnitude, contact] (2)
            self.force_direction_encoder = ForceComponentEncoder(3, force_hidden_dim)
            self.force_magnitude_encoder = ForceComponentEncoder(2, force_hidden_dim)
            self.input_proj_force_direction = nn.Linear(force_hidden_dim, hidden_dim)
            self.input_proj_force_magnitude = nn.Linear(force_hidden_dim, hidden_dim)
            if self.use_force_magnitude_head:
                # predicts the next num_queries magnitudes (one-shot from a single hidden state,
                # the same chunking pattern action_head itself uses for the action sequence)
                self.force_magnitude_pred_head = nn.Linear(force_hidden_dim, num_queries)

        # encoder extra parameters
        self.latent_dim = 32 # size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_state_proj = nn.Linear(state_dim, hidden_dim)  # project state to embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], state, actions

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        num_additional_tokens = 2  # latent, proprio
        if backbones is not None and self.force_dim is not None:
            num_additional_tokens += 2  # force direction, force magnitude (each its own token)
        self.additional_pos_embed = nn.Embedding(num_additional_tokens, hidden_dim) # learned position embedding for latent, proprio, (force direction, force magnitude)

    def forward(self, obs, actions=None):
        is_training = actions is not None
        state = obs['state'] if self.backbones is not None else obs
        bs = state.shape[0]

        if is_training:
            # project CLS token, state sequence, and action sequence to embedding dim
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            state_embed = self.encoder_state_proj(state) # (bs, hidden_dim)
            state_embed = torch.unsqueeze(state_embed, axis=1)  # (bs, 1, hidden_dim)
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            # concat them together to form an input to the CVAE encoder
            encoder_input = torch.cat([cls_embed, state_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+2, bs, hidden_dim)
            # no masking is applied to all parts of the CVAE encoder input
            is_pad = torch.full((bs, encoder_input.shape[0]), False).to(state.device) # False: not a padding
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
            # query CVAE encoder
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(state.device)
            latent_input = self.latent_out_proj(latent_sample)

        # CVAE decoder
        if self.backbones is not None:
            vis_data = obs['rgb']
            if "depth" in obs:
                vis_data = torch.cat([vis_data, obs['depth']], dim=2)
            num_cams = vis_data.shape[1]

            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(num_cams):
                features, pos = self.backbones[0](vis_data[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature # (batch, hidden_dim, H, W)
                pos = pos[0] # (1, hidden_dim, H, W)
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            # proprioception features (state)
            proprio_input = self.input_proj_robot_state(state)
            # optional force tokens: direction and magnitude(+contact) each get their own encoder
            # and their own decoder token (see the __init__ comment for why)
            extra_tokens = None
            force_magnitude_pred = None
            if self.force_dim is not None:
                force_direction = obs['force'][:, 0:3]
                force_magnitude = obs['force'][:, 3:5]  # [log_magnitude, contact]
                h_dir = self.force_direction_encoder(force_direction)
                h_mag = self.force_magnitude_encoder(force_magnitude)
                force_direction_input = self.input_proj_force_direction(h_dir)
                force_magnitude_input = self.input_proj_force_magnitude(h_mag)
                extra_tokens = [force_direction_input, force_magnitude_input]
                if self.use_force_magnitude_head:
                    force_magnitude_pred = self.force_magnitude_pred_head(h_mag).unsqueeze(-1) # (batch, num_queries, 1)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3) # (batch, hidden_dim, 4, 8)
            pos = torch.cat(all_cam_pos, axis=3) # (batch, hidden_dim, 4, 8)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, extra_tokens=extra_tokens)[0] # (batch, num_queries, hidden_dim)
        else:
            state = self.input_proj_robot_state(state)
            force_magnitude_pred = None
            hs = self.transformer(None, None, self.query_embed.weight, None, latent_input, state, self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar], force_magnitude_pred


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder
