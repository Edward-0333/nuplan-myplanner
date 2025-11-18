import time
from typing import List, Optional, Type

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.training.preprocessing.feature_builders.common import rotate_round_z_axis

from .planner_utils import global_trajectory_to_states, load_checkpoint


class ImitationPlanner(AbstractPlanner):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: str = None,
        replan_interval: int = 1,
        use_gpu: bool = True,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        """
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt
        self._initialization: Optional[PlannerInitialization] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1

        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._planner = self._planner.to(self.device)
        self._initialization = initialization

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _planning(self, current_input: PlannerInput):
        self._start_time = time.perf_counter()
        planner_feature = self._planner_feature_builder.get_features_from_simulation(
            current_input, self._initialization
        )
        planner_feature_torch = planner_feature.collate(
            [planner_feature.to_feature_tensor().to_device(self.device)]
        )
        self._feature_building_runtimes.append(time.perf_counter() - self._start_time)

        out = self._planner.forward(planner_feature_torch.data)
        print(1)
        # local_trajectory = out["output_trajectory"][0].cpu().numpy()
        self.test_plotting(out, planner_feature_torch.data)
        # return local_trajectory.astype(np.float64)

    def test_plotting(self,out_data, input_data):
        plot_agent = 14
        target_lane_probs = out_data['target_lane_probs'].cpu().numpy()[0]
        map_data = input_data['map']
        agent_data = input_data['agent']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(50, 50))
        # 绘制车辆位置
        agent_positions = agent_data['position'][0,: , -1, :2].cpu().numpy()
        agent_headings = agent_data['heading'][0,: ,-1].cpu().numpy()
        agent_shapes = agent_data['shape'][0,:,-1].cpu().numpy()
        agent_categories = agent_data['category'][0,:].cpu().numpy()
        if agent_categories[plot_agent] == 2 or agent_categories[plot_agent] ==3:
            return 0
        for i in range(agent_positions.shape[0]):

            rear_x  = agent_positions[i,0]
            rear_y  = agent_positions[i,1]
            L = agent_shapes[i,1]
            W = agent_shapes[i,0]
            theta = agent_headings[i]
            local_corners = np.array([
                [L / 2, W / 2],
                [L / 2, -W / 2],
                [-L / 2, -W / 2],
                [-L / 2, W / 2]
            ])
            # 旋转矩阵
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            global_corners = (local_corners @ R.T) + np.array([rear_x, rear_y])
            if i == 0:
                color = 'red'
            else:
                color = 'gray'
            if i == plot_agent:
                color = 'yellow'
            ax.fill(global_corners[:, 0], global_corners[:, 1],
                    facecolor=color, edgecolor='k', alpha=0.5, linewidth=2)

        # #绘制车道线
        positions = map_data['point_position_raw'][0].cpu().numpy()  # (M, 3, 21, 2)
        for m in range(positions.shape[0]):
            polygon_points = np.vstack([positions[m, 1, :, :], positions[m, 2, :, :][::-1]])
            ax.plot(positions[m, 0, :, 0], positions[m, 0, :, 1], label='Ego', linestyle='dashed', color='grey')
            ax.fill(polygon_points[:,0], polygon_points[:,1],
                     color='lightgrey', alpha=0.1,edgecolor='black',linewidth=2)
        # 绘制预测车道
        pre_data_for_agent = target_lane_probs[plot_agent] # [t, M]
        pre_lane_id = np.full((pre_data_for_agent.shape[0],), -1, dtype=np.int64)
        for t in range(pre_data_for_agent.shape[0]):
            lane_probs = pre_data_for_agent[t]
            topk_indices = lane_probs.argsort()[-3:][::-1]
            topk_vals = lane_probs[topk_indices]
            pre_lane_data = positions[topk_indices]  # (3, 3, 21, 2)
            prob_min = float(topk_vals.min())
            prob_max = float(topk_vals.max())
            prob_range = prob_max - prob_min
            # for lane_idx in range(pre_lane_data.shape[0]):
            #     polygon_points = np.vstack([pre_lane_data[lane_idx, 1, :, :], pre_lane_data[lane_idx, 2, :, :][::-1]])
            #     if prob_range > 1e-6:
            #         normalized_prob = (topk_vals[lane_idx] - prob_min) / prob_range
            #     else:
            #         normalized_prob = 0.5
            #     alpha = 0.2 + 0.6 * np.clip(normalized_prob, 0.0, 1.0)
            #     ax.fill(polygon_points[:, 0], polygon_points[:, 1],
            #             color='blue', alpha=alpha, edgecolor='black', linewidth=2,zorder=-5)
            if prob_max > 0.5:
                pre_lane_id[t] = topk_indices[0]
        pre_lane_id_unique = np.unique(pre_lane_id)
        for pre in pre_lane_id_unique:
            if pre != -1:
                pre_lane_data = positions[pre]
                polygon_points = np.vstack([pre_lane_data[1, :, :], pre_lane_data[2, :, :][::-1]])
                ax.fill(polygon_points[:, 0], polygon_points[:, 1],
                        color='blue', alpha=0.5, edgecolor='black', linewidth=2,zorder=-5)

        ax.axis('equal')
        plt.show()
        print(1)

        pass

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        ego_state = current_input.history.ego_states[-1]

        if self._last_plan_elapsed_step >= self._replan_interval:
            local_trajectory = self._planning(current_input)
            self._global_trajectory = self._get_global_trajectory(
                local_trajectory, ego_state
            )
            self._last_plan_elapsed_step = 0
        else:
            self._global_trajectory = self._global_trajectory[1:]

        trajectory = InterpolatedTrajectory(
            trajectory=global_trajectory_to_states(
                global_trajectory=self._global_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(self._global_trajectory) * self._step_interval,
                step_interval=self._step_interval,
            )
        )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)

        self._last_plan_elapsed_step += 1

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory
