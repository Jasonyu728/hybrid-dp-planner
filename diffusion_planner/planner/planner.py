
import warnings
import os
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

def identity(ego_state, predictions):
    return predictions


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer
        self._trajectory_debug_log_path = getattr(config, "trajectory_debug_log_path", None)
        self._trajectory_debug_step = 0
        self._trajectory_output_mode = getattr(config, "trajectory_output_mode", "token")
        self._use_continuous_head = getattr(config, "use_continuous_head", False)
        self._token_selection_mode = getattr(config, "token_selection_mode", "nearest")
        self._use_token_classifier = getattr(config, "use_token_classifier", False)

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            ckpt: Dict = torch.load(self._ckpt_path, map_location=self._device)
            ckpt_epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
            state_dict = ckpt
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # Strip DDP "module." prefix when present; fall back to raw keys if not.
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            if not model_state_dict:
                model_state_dict = state_dict
            self._planner.load_state_dict(model_state_dict)
            print(
                "[DiffusionPlanner] "
                f"ckpt={self._ckpt_path}, epoch={ckpt_epoch}, ema={self._ema_enabled}, "
                f"trajectory_output_mode={self._trajectory_output_mode}, "
                f"use_continuous_head={self._use_continuous_head}, "
                f"token_selection_mode={self._token_selection_mode}, "
                f"use_token_classifier={self._use_token_classifier}"
            )
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:

        raw_predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(raw_predictions[:, 3], raw_predictions[:, 2])[..., None]
        predictions = np.concatenate([raw_predictions[..., :2], heading], axis=-1)
        predictions, safety_applied, safety_reason = self._sanitize_prediction_xyh(predictions)
        self._log_trajectory_debug(raw_predictions, predictions, safety_applied, safety_reason)

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states

    def _sanitize_prediction_xyh(self, predictions: np.ndarray):
        """
        极简 sanitization：只在模型输出真的会让 nuplan 下游 SVD 崩时介入。
        其它情况完全 pass-through，零静默修改。
        """
        predictions = predictions.astype(np.float64, copy=True)
        safety_reasons = []

        if not np.isfinite(predictions).all():
            # SVD-killer: NaN/Inf。直接 nan_to_num 即可（无穷大替成 0），下游不会再崩。
            safety_reasons.append("nonfinite")
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

        return predictions, bool(safety_reasons), "|".join(safety_reasons)

    def _log_trajectory_debug(
        self,
        raw_predictions: np.ndarray,
        predictions: np.ndarray,
        safety_applied: bool = False,
        safety_reason: str = "",
    ) -> None:
        if not self._trajectory_debug_log_path:
            return

        log_dir = os.path.dirname(self._trajectory_debug_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        xy = predictions[:, :2]
        anchors = np.concatenate([np.zeros((1, 2), dtype=np.float64), xy], axis=0)
        step = np.linalg.norm(np.diff(anchors, axis=0), axis=1)
        heading = predictions[:, 2]
        heading_jump = np.abs(np.diff(np.unwrap(heading))) if len(heading) > 1 else np.array([0.0])
        cos_sin_norm = np.linalg.norm(raw_predictions[:, 2:4], axis=1)

        values = {
            "step": self._trajectory_debug_step,
            "safety_applied": int(safety_applied),
            "safety_reason": safety_reason,
            "finite_raw": int(np.isfinite(raw_predictions).all()),
            "finite_xyh": int(np.isfinite(predictions).all()),
            "end_x": float(xy[-1, 0]),
            "end_y": float(xy[-1, 1]),
            "min_x": float(np.nanmin(xy[:, 0])),
            "max_x": float(np.nanmax(xy[:, 0])),
            "min_y": float(np.nanmin(xy[:, 1])),
            "max_y": float(np.nanmax(xy[:, 1])),
            "step_min": float(np.nanmin(step)),
            "step_max": float(np.nanmax(step)),
            "step_mean": float(np.nanmean(step)),
            "zero_step_ratio": float(np.mean(step < 1e-4)),
            "heading_jump_max": float(np.nanmax(heading_jump)),
            "cossin_norm_min": float(np.nanmin(cos_sin_norm)),
            "cossin_norm_max": float(np.nanmax(cos_sin_norm)),
        }

        write_header = not os.path.exists(self._trajectory_debug_log_path)
        with open(self._trajectory_debug_log_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(values.keys()) + "\n")
            f.write(",".join(str(v) for v in values.values()) + "\n")

        self._trajectory_debug_step += 1

    def __getstate__(self):
        # _map_api contains a database connection (not picklable); exclude it so
        # SimulationLogCallback can serialize this planner via pickle.
        state = self.__dict__.copy()
        state.pop('_map_api', None)
        state.pop('_initialization', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._map_api = None
        self._initialization = None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)
        with torch.inference_mode():
            _, outputs = self._planner(inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
