from src.visualization.tutorial_utils import visualize_nuplan_scenarios_local


import os

os.environ['NUPLAN_MAPS_ROOT']="/home/vci-4/LK/pluto/nuplan-devkit/nuplan/dataset/maps"
os.environ['NUPLAN_DATA_ROOT']="/home/vci-4/LK/pluto/nuplan-devkit/nuplan/dataset"
os.environ['NUPLAN_EXP_ROOT']="/home/vci-4/LK/pluto/nuplan-devkit/nuplan/exp"

NUPLAN_DATA_ROOT = os.getenv('NUPLAN_DATA_ROOT', '/data/sets/nuplan')
NUPLAN_MAPS_ROOT = os.getenv('NUPLAN_MAPS_ROOT', '/data/sets/nuplan/maps')
NUPLAN_DB_FILES = os.getenv('NUPLAN_DB_FILES', '/home/vci-4/LK/pluto/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits/mini')
NUPLAN_MAP_VERSION = os.getenv('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')

visualize_nuplan_scenarios_local(
    data_root=NUPLAN_DATA_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_root=NUPLAN_MAPS_ROOT,
    map_version=NUPLAN_MAP_VERSION,
    bokeh_port=8899  # This controls the port bokeh uses when generating the visualization -- if you are running
                     # the notebook on a remote instance, you'll need to make sure to port-forward it.
)