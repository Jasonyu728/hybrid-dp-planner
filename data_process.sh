###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit/nuplan/dataset/nuplan-v1.1/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/data3/yuzhuoyi/AD/DiffusionPlanner/nuplan-devkit/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/data3/yuzhuoyi/AD/DiffusionPlanner/Diffusion-Planner/nuplan_diffusionplanner_large" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 50000 \

