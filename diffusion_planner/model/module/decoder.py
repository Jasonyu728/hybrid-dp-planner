import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.token_trajectory_decoder import TokenTrajectoryDecoder
from diffusion_planner.utils.diff_decode import _differentiable_decode, smooth_trajectory_xyh
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()
        self._k_tokens = 16          # 16 motion tokens per trajectory (8s @ 0.5s/token)
        self._token_emb_dim = config.token_emb_dim
        self._learnable_token_emb = self._cfg_bool(config, "learnable_token_emb", False)
        self._token_selection_mode = self._cfg_str(config, "token_selection_mode", "nearest")
        self._use_token_classifier = bool(
            self._cfg_bool(config, "use_token_classifier", False)
            or self._token_selection_mode == "classifier"
        )
        self._trajectory_output_mode = self._cfg_str(config, "trajectory_output_mode", "token")
        self._use_continuous_head = self._cfg_bool(
            config,
            "use_continuous_head",
            self._trajectory_output_mode == "continuous_head",
        )
        if self._trajectory_output_mode == "continuous":
            self._trajectory_output_mode = "continuous_head"
            self._use_continuous_head = True

        # neighbor 专用词表路径（若未配置则退回 ego 词表，向后兼容）
        nbr_vocab_path = getattr(config, 'nbr_vocab_path', None) or config.vocab_path

        # ── Ego token embedding table ────────────────────────────────────────
        # ego 专用词表，固定不训练
        ego_vocab_size, self.ego_token_emb = self._build_token_emb(
            config.vocab_path,
            config.token_emb_dim,
            seed=42,
            learnable=self._learnable_token_emb,
        )
        self._ego_vocab_size = ego_vocab_size

        # ── Neighbor token embedding table ──────────────────────────────────
        # neighbor 专用词表（可与 ego 不同大小）
        # Neighbor token embedding table.
        nbr_vocab_size, self.nbr_token_emb = self._build_token_emb(
            nbr_vocab_path,
            config.token_emb_dim,
            seed=43,
            learnable=self._learnable_token_emb,
        )
        self._nbr_vocab_size = nbr_vocab_size
        if self._use_token_classifier:
            self.ego_token_classifier = nn.Linear(config.token_emb_dim, ego_vocab_size)
            self.nbr_token_classifier = nn.Linear(config.token_emb_dim, nbr_vocab_size)
            self._init_token_classifier(self.ego_token_classifier, self.ego_token_emb.weight[3:])
            self._init_token_classifier(self.nbr_token_classifier, self.nbr_token_emb.weight[3:])
        else:
            self.ego_token_classifier = None
            self.nbr_token_classifier = None

        if self._use_continuous_head:
            latent_dim = self._k_tokens * config.token_emb_dim
            head_hidden_dim = int(getattr(config, "continuous_head_hidden_dim", 512) or 512)
            self.continuous_head = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, head_hidden_dim),
                nn.GELU(),
                nn.Linear(head_hidden_dim, self._future_len * 4),
            )
        else:
            self.continuous_head = None

        # ── Trajectory decoders for inference ───────────────────────────────
        self.ego_traj_decoder = TokenTrajectoryDecoder(config.vocab_path)
        self.nbr_traj_decoder = TokenTrajectoryDecoder(nbr_vocab_path)

        self.dit = DiT(
            sde=self._sde,
            route_encoder = RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate, hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            output_dim=self._k_tokens * config.token_emb_dim,  # 16 * D
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type
        )

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        self._guidance_fn = config.guidance_fn
        self._guidance_scale = 2.0
        self._ego_progress_rerank_topk = max(1, int(getattr(config, "ego_progress_rerank_topk", 1)))
        self._ego_progress_rerank_beta = float(getattr(config, "ego_progress_rerank_beta", 0.0))
        self._token_debug_log_path = getattr(config, "token_debug_log_path", None)
        self._token_debug_step = 0
        self._token_decode_mode = self._cfg_str(config, "token_decode_mode", "hard")
        self._token_decode_temperature = float(getattr(config, "token_decode_temperature", 1.0))

    @staticmethod
    def _cfg_bool(config, name: str, default: bool = False) -> bool:
        value = getattr(config, name, default)
        if value is None:
            return default
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y", "t")
        return bool(value)

    @staticmethod
    def _cfg_str(config, name: str, default: str) -> str:
        value = getattr(config, name, default)
        if value is None or value == "":
            return default
        return str(value).lower()

    @staticmethod
    def _build_token_emb(
        vocab_path: str,
        token_emb_dim: int,
        seed: int = 42,
        learnable: bool = False,
    ):
        """
        从 vocab_path 加载词表，构建可训练 embedding 表（VQ-VAE 风格）。

        初始化用 centroid 随机投影提供物理先验；训练时通过 commitment loss
        让 embedding 自适应到 DiT 容易预测的位置。

        Returns
        -------
        vocab_size : int
        token_emb  : nn.Embedding  (vocab_size + 3, token_emb_dim)，requires_grad=True
        """
        _data = np.load(vocab_path, allow_pickle=True)
        _centroids = torch.tensor(_data['centroids'], dtype=torch.float32)
        vocab_size = _centroids.shape[0]
        _seg_dim   = _centroids.shape[1]

        torch.manual_seed(seed)
        _proj      = torch.randn(_seg_dim, token_emb_dim) / (_seg_dim ** 0.5)
        _motion_emb = _centroids @ _proj                                     # (V, D)
        _motion_emb = (_motion_emb - _motion_emb.mean()) / (_motion_emb.std() + 1e-6)

        emb = nn.Embedding(vocab_size + 3, token_emb_dim)
        with torch.no_grad():
            nn.init.normal_(emb.weight, mean=0.0, std=1.0)
            emb.weight[3:] = _motion_emb

        # 可训练（与原版相反）。训练循环负责用 stop-gradient + commitment loss
        # 防止 embedding 与 DiT 互相塌缩。
        for p in emb.parameters():
            p.requires_grad = learnable

        return vocab_size, emb

    @staticmethod
    def _init_token_classifier(classifier, token_emb_w):
        with torch.no_grad():
            classifier.weight.copy_(token_emb_w)
            classifier.bias.zero_()

    @property
    def sde(self):
        return self._sde

    def _select_ego_token_ids(self, x0_ego, ego_emb_w):
        dist = torch.cdist(x0_ego, ego_emb_w)
        topk = min(self._ego_progress_rerank_topk, dist.shape[-1])

        if topk <= 1 or self._ego_progress_rerank_beta <= 0.0:
            return dist.argmin(dim=-1)

        top_dist, top_ids = dist.topk(k=topk, dim=-1, largest=False)
        progress_col = 12 if self.ego_traj_decoder.centroids.shape[1] >= 15 else 0
        progress = self.ego_traj_decoder.centroids[:, progress_col].to(
            device=top_dist.device,
            dtype=top_dist.dtype,
        ).clamp(min=0.0)
        top_progress = progress[top_ids]

        rerank_score = top_dist - self._ego_progress_rerank_beta * top_progress
        selected = rerank_score.argmin(dim=-1, keepdim=True)
        return top_ids.gather(dim=-1, index=selected).squeeze(-1)

    def _log_ego_tokens(self, ego_ids, ego_future):
        if not self._token_debug_log_path:
            return

        log_dir = os.path.dirname(self._token_debug_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        progress_col = 12 if self.ego_traj_decoder.centroids.shape[1] >= 15 else 0
        raw_ids = (ego_ids - 3).clamp(0, self.ego_traj_decoder.centroids.shape[0] - 1)
        token_dx = self.ego_traj_decoder.centroids[:, progress_col][raw_ids].detach().cpu()
        token_ids_cpu = ego_ids.detach().cpu()
        endpoint_xy = ego_future[:, -1, :2].detach().cpu()

        write_header = not os.path.exists(self._token_debug_log_path)
        with open(self._token_debug_log_path, "a", encoding="utf-8") as f:
            if write_header:
                id_cols = ",".join([f"token_{i}" for i in range(self._k_tokens)])
                dx_cols = ",".join([f"dx_{i}" for i in range(self._k_tokens)])
                f.write(
                    "step,batch_index,total_token_dx,slow_token_ratio,end_x,end_y,"
                    f"{id_cols},{dx_cols}\n"
                )

            for batch_idx in range(token_ids_cpu.shape[0]):
                dx = token_dx[batch_idx]
                slow_ratio = (dx < 0.2).float().mean().item()
                ids_str = ",".join(str(int(v)) for v in token_ids_cpu[batch_idx].tolist())
                dx_str = ",".join(f"{float(v):.4f}" for v in dx.tolist())
                f.write(
                    f"{self._token_debug_step},{batch_idx},{float(dx.sum()):.4f},"
                    f"{slow_ratio:.4f},{float(endpoint_xy[batch_idx, 0]):.4f},"
                    f"{float(endpoint_xy[batch_idx, 1]):.4f},{ids_str},{dx_str}\n"
                )

        self._token_debug_step += 1

    def continuous_from_latent(self, x_latent):
        if self.continuous_head is None:
            raise RuntimeError("continuous_from_latent requires use_continuous_head=True")

        bsz, n_agents, _ = x_latent.shape
        traj = self.continuous_head(
            x_latent.reshape(bsz * n_agents, -1)
        ).reshape(bsz, n_agents, self._future_len, 4)

        direction = traj[..., 2:4]
        direction_norm = direction.norm(dim=-1, keepdim=True)
        default_direction = torch.zeros_like(direction)
        default_direction[..., 0] = 1.0
        direction = torch.where(
            direction_norm > 1e-4,
            F.normalize(direction, dim=-1, eps=1e-6),
            default_direction,
        )

        return torch.cat([traj[..., :2], direction], dim=-1)

    def _soft_decode_token_embeddings(self, x_tokens, emb_w, centroids):
        traj_xyh = _differentiable_decode(
            x_tokens,
            emb_w,
            centroids,
            tau=self._token_decode_temperature,
        )
        traj_xyh = smooth_trajectory_xyh(traj_xyh)

        traj = torch.zeros(
            x_tokens.shape[0],
            self._future_len,
            4,
            device=x_tokens.device,
            dtype=traj_xyh.dtype,
        )
        traj[:, :, 0] = traj_xyh[:, :, 0]
        traj[:, :, 1] = traj_xyh[:, :, 1]
        traj[:, :, 2] = torch.cos(traj_xyh[:, :, 2])
        traj[:, :, 3] = torch.sin(traj_xyh[:, :, 2])
        return traj

    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_current_state": current ego states,            
                    "neighbor_agent_past": past and current neighbor states,  

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :self._predicted_neighbor_num, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current], dim=1) # [B, P, 4]

        B, P, _ = current_states.shape
        assert P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']
        route_lanes = inputs['route_lanes']

        if self.training:
            # sampled_trajectories: (B, P, 16*D) — noisy token embeddings
            sampled_trajectories = inputs['sampled_trajectories'].reshape(B, P, -1)
            diffusion_time = inputs['diffusion_time']

            return {
                "score": self.dit(
                    sampled_trajectories,
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    neighbor_current_mask
                ).reshape(B, P, -1)  # (B, P, 16*D)
            }
        else:
            # xT: pure noise in token embedding space (B, P, 16*D)
            # std=1.0 matches the N(0,1) normalized embedding targets used in training
            xT = torch.randn(B, P, self._k_tokens * self._token_emb_dim,
                             device=current_states.device)

            x0 = dpm_sampler(
                        self.dit,
                        xT,
                        other_model_params={
                            "cross_c": ego_neighbor_encoding,
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask
                        },
                        dpm_solver_params={},  # no initial-state constraint in embedding space
                        model_wrapper_params={
                            "classifier_fn": self._guidance_fn,
                            "classifier_kwargs": {
                                "model": self.dit,
                                "model_condition": {
                                    "cross_c": ego_neighbor_encoding,
                                    "route_lanes": route_lanes,
                                    "neighbor_current_mask": neighbor_current_mask
                                },
                                "inputs": inputs,
                                "observation_normalizer": self._observation_normalizer,
                                "state_normalizer": self._state_normalizer,
                                # Token guidance: embedding tables and centroids
                                "ego_emb_w":     self.ego_token_emb.weight[3:].detach(),
                                "nbr_emb_w":     self.nbr_token_emb.weight[3:].detach(),
                                "ego_centroids": self.ego_traj_decoder.centroids.detach(),
                                "nbr_centroids": self.nbr_traj_decoder.centroids.detach(),
                            },
                            "guidance_scale": self._guidance_scale,
                            "guidance_type": "classifier" if self._guidance_fn is not None else "uncond"
                        },
                )
            # x0: (B, P, 16*D) — continuous embeddings output by DiT
            D   = self._token_emb_dim
            K   = self._k_tokens
            P1  = P - 1

            if self._trajectory_output_mode == "continuous_head":
                return {
                    "prediction": self.continuous_from_latent(x0)
                }

            # ── Nearest-neighbor lookup: ego ────────────────────────────────
            x0_ego    = x0[:, 0].reshape(B * K, D)                          # (B*16, D)
            ego_emb_w = self.ego_token_emb.weight[3:]                       # (V_ego, D)
            if self._token_selection_mode == "classifier" and self.ego_token_classifier is not None:
                ego_ids = self.ego_token_classifier(x0_ego).argmax(dim=-1).reshape(B, K) + 3
            else:
                ego_ids = self._select_ego_token_ids(x0_ego, ego_emb_w).reshape(B, K) + 3

            # ── Nearest-neighbor lookup: neighbor ───────────────────────────
            x0_nbr    = x0[:, 1:].reshape(B * P1 * K, D)                   # (B*P1*16, D)
            nbr_emb_w = self.nbr_token_emb.weight[3:]                       # (V_nbr, D)
            if self._token_selection_mode == "classifier" and self.nbr_token_classifier is not None:
                nbr_ids = self.nbr_token_classifier(x0_nbr).argmax(dim=-1).reshape(B, P1, K) + 3
            else:
                nbr_ids = torch.cdist(x0_nbr, nbr_emb_w).argmin(dim=-1).reshape(B, P1, K) + 3

            # ── Add BOS=1 and EOS=2 ─────────────────────────────────────────
            bos_1  = torch.ones (B,      1,  dtype=torch.long, device=x0.device)
            eos_1  = torch.full ((B,      1), 2, dtype=torch.long, device=x0.device)
            bos_P1 = torch.ones (B, P1,  1,  dtype=torch.long, device=x0.device)
            eos_P1 = torch.full ((B, P1,  1), 2, dtype=torch.long, device=x0.device)

            ego_full_ids = torch.cat([bos_1,  ego_ids,               eos_1 ], dim=1)  # (B, 18)
            nbr_full_ids = torch.cat([bos_P1, nbr_ids, eos_P1], dim=2)                # (B, P1, 18)

            # ── Decode token IDs → continuous trajectories ───────────────────
            if self._token_decode_mode == "soft":
                ego_future = self._soft_decode_token_embeddings(
                    x0[:, 0].reshape(B, K, D),
                    ego_emb_w,
                    self.ego_traj_decoder.centroids,
                )
            else:
                ego_future = self.ego_traj_decoder.decode_tokens(ego_full_ids)   # (B, 80, 4)
            self._log_ego_tokens(ego_ids, ego_future)
            nbr_future = self.nbr_traj_decoder.decode_tokens(
                nbr_full_ids.reshape(B * P1, 18)
            ).reshape(B, P1, self._future_len, 4)                            # (B, P1, 80, 4)

            prediction = torch.cat([ego_future.unsqueeze(1), nbr_future], dim=1)  # (B, P, 80, 4)

            return {
                    "prediction": prediction
                }

        
class RouteEncoder(nn.Module):
    def __init__(self, route_num, lane_len, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device, dtype=x.dtype)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1, mlp_ratio=4.0, model_type="x_start"):
        super().__init__()
        
        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=max(512, output_dim // 2), out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std
               
    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask):
        """
        Forward pass of DiT.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, P, _ = x.shape
        
        x = self.preproj(x)

        x_embedding = torch.cat([self.agent_embedding.weight[0][None, :], self.agent_embedding.weight[1][None, :].expand(P - 1, -1)], dim=0)  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1) # (B, P, D)
        x = x + x_embedding     

        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        y = y + self.t_embedder(t)      

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask
        
        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)  
            
        x = self.final_layer(x, y)
        
        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
