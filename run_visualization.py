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
    # 创建文件夹
    # '/home/vci-4/LK/lk_nuplan/my_planner/scenario_pngs'
    save_path = f'/home/vci-4/LK/lk_nuplan/my_planner/scenario_pngs/{scenario_name[0]}/{scenario_name[1]}/{scenario_name[2]}'
    # 如果文件夹不存在，则创建
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    map_data = data['map']
    positions = map_data['point_position_raw']  # (N, 3, 21, 2)
    polygon_type = map_data['polygon_type']  # (N,)
    polygon_on_route = map_data['polygon_on_route']  # (N,)
    polygon_road_block_id = map_data['polygon_road_block_id']  # (N,)
    polygon_road_block_id_set = set(polygon_road_block_id)
    cand_mask = data['agent']['cand_mask']
    dict_all_lane_id = data['agent']['dict_all_lane_id']
    # 设置一个列表，存储每个road_block_id对应的颜色
    road_block_id_color = {}
    for road_block_id in polygon_road_block_id_set:
        road_block_id_color[road_block_id] = (random.random(), random.random(), random.random())
    polygon_on_route_id = map_data['polygon_road_lane_id'][polygon_on_route]
    # target_lane_matrix与polygon_on_route_id一一对应，提前建立lane_id到权重的映射
    lane_id_to_target_weight: Dict[Union[int, str], float] = {}
    # if target_lane_matrix.size:
    #     target_array = np.atleast_2d(np.asarray(target_lane_matrix))
    #     flattened_target = target_array[0].reshape(-1)
    #     for lane_idx, lane_id in enumerate(polygon_on_route_id):
    #         if lane_idx < flattened_target.size:
    #             lane_id_to_target_weight[lane_id] = flattened_target[lane_idx]
    agent = 2
    for t in range(101):
        fig, ax = plt.subplots(figsize=(50, 50))
        now_lane = data['agent']['lane_id'][agent][t]
        now_block = data['agent']['roadblock_id'][agent][t]
        # plot map
        N = positions.shape[0]
        print('正在绘制场景:', scenario_name, '时间步:', t, '多边形数量:', N)
        for i in range(N):
            road_lane_id = map_data['polygon_road_lane_id'][i]
            if road_lane_id == 0:
                cand_lane = False
            else:
                lane_idx = dict_all_lane_id[road_lane_id]
                cand_lane = cand_mask[agent,t,lane_idx]
            polygon_points = np.vstack([positions[i,1,:,:], positions[i,2,:,:][::-1]])
            ax.plot(positions[i,0,:,0], positions[i,0,:,1], label='Ego',linestyle='dashed',color='grey')

            if cand_lane:
                color = 'orange'
            else:
                color = 'lightblue'

            # if polygon_type[i] == 0:
            # elif polygon_type[i] == 2:
            #     color = 'lightsteelblue'
            # elif polygon_type[i] == 1:
            #     color = 'cyan'
            # target_val = lane_id_to_target_weight.get(road_lane_id)
            # if target_val is not None and target_val > 0:
            #     val = float(np.clip(target_val, 0.0, 1.0))
            #     color = plt.cm.get_cmap('Greens')(0.2 + 0.8 * val)
            # polygon_road_block_id_i = polygon_road_block_id[i]
            # 绘制同一road_block_id的多边形使用相同颜色
            # color = road_block_id_color[polygon_road_block_id_i]
            # if polygon_on_route[i] == 1:
            #     color = 'yellow'
            # if road_lane_id == int(now_lane) and road_lane_id != 0:
            #     color = 'green'
            ax.fill(polygon_points[:,0], polygon_points[:,1],
                     color=color, alpha=0.5,edgecolor='black',linewidth=2)

        # plot ego car
        agent_data = data['agent']
        ego_parameters = get_pacifica_parameters()
        rear_x  = agent_data['position'][0,t,0]
        rear_y  = agent_data['position'][0,t,1]
        offset = ego_parameters.rear_axle_to_center
        L = agent_data['shape'][0,t,1]
        W = agent_data['shape'][0,t,0]
        theta = agent_data['heading'][0,t]
        # 计算车辆中心
        cx = rear_x + offset * np.cos(theta)
        cy = rear_y + offset * np.sin(theta)
        # 局部坐标系下的矩形顶点 (逆时针)
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
        # 全局坐标
        global_corners = (local_corners @ R.T) + np.array([cx, cy])
        # 绘制矩形
        ax.fill(global_corners[:, 0], global_corners[:, 1],
                facecolor='red', edgecolor='k', alpha=0.5, linewidth=2)

        # 绘制其他车辆
        num_agents = agent_data['position'].shape[0]
        for i in range(1, num_agents):
            if not agent_data['valid_mask'][i,t]:
                continue

            agent_block = agent_data['roadblock_id'][i,t]
            # color = road_block_id_color[agent_block]
            rear_x  = agent_data['position'][i,t,0]
            rear_y  = agent_data['position'][i,t,1]
            L = agent_data['shape'][i,t,1]
            W = agent_data['shape'][i,t,0]
            theta = agent_data['heading'][i,t]
            # 局部坐标系下的矩形顶点 (逆时针)
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
            # 全局坐标
            global_corners = (local_corners @ R.T) + np.array([rear_x, rear_y])
            if agent_data['in_route_block'][i,t]:
                color = 'gold'
            else:
                color = 'grey'
            if i == agent:
                color = 'blue'
            # 绘制矩形
            ax.fill(global_corners[:, 0], global_corners[:, 1],
                    facecolor=color, edgecolor='k', alpha=0.5, linewidth=2)
        ax.axis('equal')
        plt.savefig(f'{save_path}/{t:03d}.png')
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
        # file_token = file_name.split('/')[-2]
        # if not file_token=="b2a5c363d1dd5abe":
        #     continue
        # if j <=6:
        #     j += 1
        #     continue
        scenario_name = file_name.split('/')[-4:-1]
        feature = storing_mechanism.load_computed_feature_from_folder(pathlib.Path(file_name), feature_builders.get_feature_type())
        plot_scenarios(feature.data,scenario_name)
        j += 1
        if j == 20:
            break


if __name__ == '__main__':
    main()
