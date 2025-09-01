from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter

from typing import Dict, List, Optional, Union
import hydra
import os
from omegaconf import DictConfig
import pathlib

CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', './config')
CONFIG_NAME = 'default_training'
os.environ['NUPLAN_MAPS_ROOT']="/home/vci-4/LK/lk_nuplan/nuplan-devkit/nuplan/dataset/maps"
os.environ['NUPLAN_DATA_ROOT']="/home/vci-4/LK/lk_nuplan/nuplan-devkit/nuplan/dataset"
os.environ['NUPLAN_EXP_ROOT']="/home/vci-4/LK/lk_nuplan/nuplan-devkit/nuplan/exp"


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
        for root, dirs, files in os.walk(cache_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if os.path.isdir(dir_path):
                    for sub_root, sub_dirs, sub_files in os.walk(dir_path):
                        for sub_dir in sub_dirs:
                            sub_dir_path = os.path.join(sub_root, sub_dir)
                            if os.path.isdir(sub_dir_path):
                                for sub_sub_root, sub_sub_dirs, sub_sub_files in os.walk(sub_dir_path):
                                    for file in sub_sub_files:
                                        if file == feature_builders.get_feature_unique_name() + '.gz':
                                            file_path = os.path.join(sub_sub_root, file)
                                            file_names.append(file_path)

    for file_name in file_names:

        feature = storing_mechanism.load_computed_feature_from_folder(pathlib.Path(file_name), feature_builders.get_feature_type())

        print(file_name)

    print(1)


if __name__ == '__main__':
    main()

