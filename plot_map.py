import random

from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from typing import Dict, List, Optional, Union
import hydra
import os
from omegaconf import DictConfig
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', './config')
CONFIG_NAME = 'default_training'
os.environ['NUPLAN_MAPS_ROOT']="/home/vci-4/LK/lk_nuplan/nuplan-devkit/nuplan/dataset/maps"
os.environ['NUPLAN_DATA_ROOT']="/home/vci-4/LK/lk_nuplan/nuplan-devkit/nuplan/dataset"
os.environ['NUPLAN_EXP_ROOT']="/home/vci-4/LK/lk_nuplan/nuplan-devkit/nuplan/exp"

def plot_scenarios(data,scenario_name):
    lane_cand_valid = data['agent']['lane_cand_valid'][:,21:]
    agent_lane_id_target = data['agent']['agent_lane_id_target']

    # 创建文件夹
    save_path = f'/home/vci-4/LK/lk_nuplan/my_planner/scenario_pngs/{scenario_name[0]}/{scenario_name[1]}/{scenario_name[2]}'
    # 如果文件夹不存在，则创建
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    map_data = data['map']
    polygon_connect_matrix = map_data['polygon_connect_matrix']  # (N, N)
    polygon_connect_cost_matrix = map_data['polygon_connect_cost_matrix']
    positions = map_data['point_position_raw']  # (N, 3, 21, 2)
    polygon_type = map_data['polygon_type']  # (N,)
    polygon_on_route = map_data['polygon_on_route']  # (N,)
    polygon_road_block_id = map_data['polygon_road_block_id']  # (N,)
    polygon_road_block_id_set = set(polygon_road_block_id)
    cand_mask = data['agent']['lane_cand_valid']
    all_lane_id = data['map']['polygon_road_lane_id']
    dict_all_lane_id = {int(lid): idx for idx, lid in enumerate(all_lane_id) if int(lid) > 0}
    # 设置一个列表，存储每个road_block_id对应的颜色
    road_block_id_color = {}
    for road_block_id in polygon_road_block_id_set:
        road_block_id_color[road_block_id] = (random.random(), random.random(), random.random())
    polygon_on_route_id = map_data['polygon_road_lane_id'][polygon_on_route]
    # target_lane_matrix与polygon_on_route_id一一对应，提前建立lane_id到权重的映射
    lane_id_to_target_weight: Dict[Union[int, str], float] = {}
    N = positions.shape[0]
    start_lane = 0
    polygon_connect_cost_matrix_i = polygon_connect_cost_matrix[start_lane]
    fig, ax = plt.subplots(figsize=(50, 50))
    cost_color_map: Dict[Union[int, float], tuple] = {}

    for i in range(N):

        road_lane_id = map_data['polygon_road_lane_id'][i]
        lane_idx = dict_all_lane_id[road_lane_id]
        cost_value = float(polygon_connect_cost_matrix_i[lane_idx])
        polygon_points = np.vstack([positions[i, 1, :, :], positions[i, 2, :, :][::-1]])
        if i == start_lane:
            ax.plot(positions[i, 0, :, 0], positions[i, 0, :, 1], label='Ego', linestyle='-', color='black', linewidth=3)
        ax.plot(positions[i, 0, :, 0], positions[i, 0, :, 1], label='Ego', linestyle='dashed', color='grey')

        base_color = cost_color_map.setdefault(cost_value, (random.random(), random.random(), random.random()))

        ax.fill(polygon_points[:, 0], polygon_points[:, 1],
                color=base_color, alpha=0.5, edgecolor='black', linewidth=2)
    ax.axis('equal')
    plt.savefig(f'{save_path}/{1}.png')
    plt.close()


def build_scenarios_from_config(
    cfg: DictConfig, scenario_builder: AbstractScenarioBuilder, worker: WorkerPool
) -> List[AbstractScenario]:
    """
    Build scenarios from config file.
    :param cfg: Omegaconf dictionary
    :param scenario_builder: Scenario builder.
    :param worker: Worker to submit tasks which can be executed in parallel
    :return: A list of scenarios
    """
    scenario_filter = build_scenario_filter(cfg.scenario_filter)
    return scenario_builder.get_scenarios(scenario_filter, worker)  # type: ignore

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    scenario_from_config = False

    cache_path = pathlib.Path(cfg.cache.cache_path)

    feature_cache = FeatureCachePickle()
    model = build_torch_module_wrapper(cfg.model)

    feature_builders = model.get_list_of_required_feature()[0]
    target_builders = model.get_list_of_computed_target()[0]

    del model
    storing_mechanism = FeatureCachePickle()
    file_names=[]
    if scenario_from_config:
        # build scenario builder
        scenario_builder = build_scenario_builder(cfg)

        # build worker
        worker = build_worker(cfg)

        scenarios = build_scenarios_from_config(cfg, scenario_builder, worker)
        print(len(scenarios))
        for scenario in scenarios:
            print(scenario.log_name, scenario.scenario_type, scenario.token)
            file_name = (
                cache_path / scenario.log_name / scenario.scenario_type / scenario.token / feature_builders.get_feature_unique_name()
            )
            file_names.append(file_name)

    else:
        # load from existing file
        # 读取cache_path下的所有文件夹
        target_name = feature_builders.get_feature_unique_name() + '.gz'

        for root, dirs, files in os.walk(cache_path):
            for file in files:
                if file == target_name:
                    file_path = os.path.join(root, file)
                    file_names.append(file_path)
    file_names = sorted(file_names)
    j = 0
    for file_name in file_names:
        file_token = file_name.split('/')[-2]
        print(file_token)
        # if not file_token=="ecc632f0d9805f47":
        #     continue
        # if j <=6:
        #     j += 1
        #     continue
        scenario_name = file_name.split('/')[-4:-1]
        feature = storing_mechanism.load_computed_feature_from_folder(pathlib.Path(file_name), feature_builders.get_feature_type())
        plot_scenarios(feature.data,scenario_name)
        j += 1
        if j == 10:
            break


if __name__ == '__main__':
    main()
