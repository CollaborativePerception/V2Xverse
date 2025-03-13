# Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System

[Paper](https://arxiv.org/pdf/2404.09496) | [Project page](https://collaborativeperception.github.io/V2Xverse/)

This repository contains the official PyTorch implementation of paper "Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System".

![V2X autonomous driving](simulation/demo/demo.gif)

## Features
Open source:
- [x] Dataset
- [x] Checkpoints


Support the developing of our CoDriving system in three tasks:
- [x] Closed-loop driving
- [x] 3D object detection
- [x] Waypoints prediction

Support the deployments of SOTA end-to-end autonomous driving methods in Carla-based benchmark:
- [ ] [WOR [ICCV 2021]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Learning_To_Drive_From_a_World_on_Rails_ICCV_2021_paper.pdf)
- [ ] [Transfuser [CVPR 2021]](https://openaccess.thecvf.com/content/CVPR2021/papers/Prakash_Multi-Modal_Fusion_Transformer_for_End-to-End_Autonomous_Driving_CVPR_2021_paper.pdf)
- [ ] [LAV [CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Learning_From_All_Vehicles_CVPR_2022_paper.pdf)
- [ ] [TCP [NIPS 2022]](https://arxiv.org/pdf/2206.08129)
- [ ] [Interfuser [CoRL 2022]](https://arxiv.org/pdf/2207.14024)

Support the complete developing pipeline (training + offline evaluation + closed-loop driving evaluation) of multiple collaborative perception methods:
- [x] Late fusion
- [x] Early fusion
- [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
- [x] [V2VNet [ECCV2022]](https://arxiv.org/abs/2008.07519)
- [ ] [DiscoNet [NeurIPS2022]](https://arxiv.org/abs/2111.00643)
- [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)
- [ ] [HEAL [ICLR2024]](https://openreview.net/forum?id=KkrDUGIASk)

Modality:
- [x] Lidar
- [ ] Camera (coming soon)



## Contents
1. [Installation](#introduction)
2. [Dataset](#dataset)
3. [Training](#train)
4. [Closed loop evaluation](#closed-loop)
5. [Modular evaluation](#modular)
6. [Shutdown simulation](#Shutdown)
7. [Todo](#Todo)
8. [Acknowledgements](#Acknowledgements)

## <span id="introduction"> Installation


### Step 1: Basic Installation
Get code and create pytorch environment.
```Shell
git clone https://github.com/CollaborativePerception/V2Xverse.git
conda create --name v2xverse python=3.7 cmake=3.22.1
conda activate v2xverse
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

cd V2Xverse
pip install -r opencood/requirements.txt
pip install -r simulation/requirements.txt
```

### Step 2: Download and setup CARLA 0.9.10.1.
```Shell
chmod +x simulation/setup_carla.sh
./simulation/setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
mkdir external_paths
ln -s ${PWD}/carla/ external_paths/carla_root
# If you already have a Carla, just create a soft link to external_paths/carla_root
```
Note: we choose the setuptools==41 to install because this version has the feature `easy_install`. After installing the carla.egg you can install the lastest setuptools to avoid No module named distutils_hack.

### Step 3: Install Spconv (1.2.1)
We use spconv 1.2.1 to generate voxel features in perception module.

To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

### Step 4: Set up opencood
```Shell
# Set up
python setup.py develop

# Bbx IOU cuda version compile
python opencood/utils/setup.py build_ext --inplace 
```

### Step 5: Install pypcd
```Shell
# go to another folder
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ..
```

### Step 6: Install EfficinetNet(required by camera detector Lift-Splat-Shoot)
```Shell
pip install efficientnet_pytorch==0.7.0
```

## <span id="dataset"> Dataset
There are two ways to obtain dataset, you can generate a dataset by youself or download one from [huggingface](https://huggingface.co/datasets/gjliu/V2Xverse), you may download dataset at the root directory of this repository.

Here are the steps to generate a dataset, where we employ a strong privileged rule-based expert agent as supervisor.

```Shell
# Generate a dataset in parallel

cd V2Xverse
# Initialize dataset directory
python ./simulation/data_collection/init_dir.py --dataset_dir  ./dataset

# Generate scripts for every routes
python ./simulation/data_collection/generate_scripts.py

# Link dataset directory, if you initialized dataset in other directory, replace ./dataset with your dataset directory
ln -s ${PWD}/dataset/ ./external_paths/data_root

# Open Carla server (15 parallel process in total)
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=40000 -prefer-nvidia
CUDA_VISIBLE_DEVICES=1 ./external_paths/carla_root/CarlaUE4.sh --world-port=40002 -prefer-nvidia
CUDA_VISIBLE_DEVICES=2 ./external_paths/carla_root/CarlaUE4.sh --world-port=40004 -prefer-nvidia
...
CUDA_VISIBLE_DEVICES=7 ./external_paths/carla_root/CarlaUE4.sh --world-port=40028 -prefer-nvidia

# Execute data generation in parallel
bash simulation/data_collection/generate_v2xverse_all.sh
```

Generate data on one single route.
```Shell
# Open one Carla server
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=40000 -prefer-nvidia

# Execute data generation for route 0 in town01
bash ./simulation/data_collection/scripts/weather-0/routes_town01_0.sh
```
Tips: set usable --world-port and adjust ${PORT} in /simulation/data_collection/scripts/weather-0/routes_townXX_X.sh accordingly. Otherwise, the python programme might stuck.

The files in dataset should follow this structure:
```Shell
|--weather-0
    |--data
        |--routes_town{town_id}_{route_id}_w{weather_id}_{datetime}
            |--ego_vehicle_{vehicle_id}
                |--2d_bbs_{direction}
                |--3d_bbs
                |--actors_data
                |--affordances
                |--bev_visibility
                |--birdview
                |--depth_{direction}
                |--env_actors_data
                |--lidar
                |--lidar_semantic_front
                |--measurements
                |--rgb_{direction}
                |--seg_{direction}
                |--topdown
            |--rsu_{vehicle_id}
            |--log
    |--results
...
|--weather-13
```

Once a new dataset is generated in `./dataset`, generate a index file with:
```Shell
python simulation/data_collection/gen_index.py
```
This will result in `dataset/dataset_index.txt`, from which we retrieval dataset sub-directory in training and testing.

## <span id="train"> Training

### Perception module
We use yaml files to configure parameters to train perception module. See `opencood/hypes_yaml/v2xverse/` for examples.

To train perception module from scratch or a continued checkpoint, run the following commonds:
```Shell
python opencood/tools/train.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `-y`: the path of the training configuration file, e.g. `opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml`, meaning you want to train the perception module of our codriving system. Using `opencood/hypes_yaml/v2xverse/fcooper_multiclass_config.yaml` means you want to train the fcooper perception model.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune or continue-training. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder. In this case, ${CONFIG_FILE} can be `None`,

Train the perception module in DDP:
```Shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch  --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]
```
`--nproc_per_node` indicate the GPU number you will use.

Test the perception module:
```Shell
python opencood/tools/inference_multiclass.py --model_dir ${CHECKPOINT_FOLDER}
```

Test the perception module in latency setting:
```Shell
python opencood/tools/inference_multiclass_latency.py --model_dir ${CHECKPOINT_FOLDER}
```

Test the perception module in pose error setting:
```Shell
python opencood/tools/inference_multiclass_w_noise.py --model_dir ${CHECKPOINT_FOLDER}
```


### Planning module
Given a checkpoint of perception module, we freeze its parameters and train the down-stream planning module ([MotionNet](https://arxiv.org/abs/2003.06754) as backbone) in an end-to-end paradigm. The planner gets BEV perception feature and occupancy map as input and targets to predict the future waypoints of ego vehicle.

Train the planning module with a given perception checkpoint:
```Shell
bash scripts/train_planner_e2e.sh ${CUDA_VISIBLE_DEVICES} ${NUM_GPUS} ${perception_model_dir} ${collaboration_method} ${planner_resume}
```
Arguments Explanation:
- `CUDA_VISIBLE_DEVICES`: ids of GPUs to be used.
- `NUM_GPUS`: number of GPUs to be used.
- `perception_model_dir` : the path of the folder that contains perception checkpoint.
- `collaboration_method` : we now support codriving/early/late/single/fcooper/v2xvit. Make sure to be consistent with the method used in `perception_model_dir`. You can adjust the corresponding configuration file in `codriving/hypes_yaml/codriving/end2end_${collaboration_method}.yaml`.
- `planner_resume` (optional): the checkpoint path for planner to resume.

Test the entire driving system (perception+planning) in waypoints prediction task with ADE and FDE:
```Shell
bash scripts/eval_planner_e2e.sh  ${CUDA_VISIBLE_DEVICES} ${perception_model_dir} ${collaboration_method} ${planner_resume}
```
This evaluation measures the ability of driving system to clone the behaviors of expert agent.

Test the waypoints prediction task in latency setting:
```Shell
bash scripts/eval_planner_e2e_latency.sh  ${CUDA_VISIBLE_DEVICES} ${perception_model_dir} ${collaboration_method} ${planner_resume}
```

Test the waypoints prediction task in pose error setting:
```Shell
bash scripts/eval_planner_e2e_w_noise.sh  ${CUDA_VISIBLE_DEVICES} ${perception_model_dir} ${collaboration_method} ${planner_resume}
```

## <span id="closed-loop"> Closed-loop evaluation
- For collaborative autonomous driving, you can set up your collaborative agents with perception and planning module, and run them in V2Xverse simulation!
- For single-agent driving, we provide the deployment of SOTA end-to-end AD methods in V2Xverse.(coming soon)

Your can customize closed-loop evaluation with specific agents and scenarios.
For evaluation on one route, following these steps:
```Shell
# Open one Carla server
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=${Carla_port} -prefer-nvidia

# Evaluation on one route
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_driving_e2e.sh ${Route_id} ${Carla_port} ${Method_tag} ${Repeat_id} ${Agent_config} ${Scenario_config}
```
Arguments Explanation:
- `Route_id`: the id of test route, corresponding to the route file `simulation/leaderboard/data/evaluation_routes/town05_short_r${route_id}.xml`. The route is defined through a sequence of waypoints in Carla town.
- `Carla_port`: the port used for python programme to communicate with Carla simulation. Make sure to be consistent with the argument `--world-port` when opening Carla server.
- `Method_tag & Repeat_id`: personalized tags for the method and this time of running, e.g. Method_tag: codriving & Repeat_id:0.
- `Agent_config`: configuration of agent, corresponding to the file `simulation/leaderboard/team_code/agent_config/pnp_config_${Agent_config}.yaml`. This file contains important features for autonomous agent, from model to PID control. Custumize your own agent by editting this file and set the inside parameters `perception_model_dir` and `planner_model_checkpoint` and `planner_config` with your own path, see an example `simulation/leaderboard/team_code/agent_config/example_config.yaml`.
- `Scenario_config`: configuration of scenario, corresponding to the file `simulation/leaderboard/leaderboard/scenarios/scenario_parameter_${Scenario_config}.yaml`. We provide five configuration files in advance.


## <span id="Checkpoints"> Checkpoints
Your can download the checkpoints from [codriving models on huggingface](https://huggingface.co/gjliu/v2xverse) and put each folder in `./checkpoints`, for example:

```Shell
|--checkpoints
    |--codriving
        |--perception
        |--planning
```

## <span id="Shutdown"> Shut down simulation on Linux
Carla processes may fail to stop，please kill them in time.

Display your processes
~~~
ps U usrname | grep PROCESS_NAME(eg. python，carla)
~~~
Kill process
~~~
kill -9 PID
~~~
Kill all carla-related processes
~~~
ps -def |grep 'carla' |cut -c 9-15| xargs kill -9
pkill -u username -f carla
~~~

## <span id="Todo"> Todo
- [x] Data generation
- [x] Training
- [x] Closed-loop evaluation
- [x] Modular evaluation
- [x] Dataset and checkpoint release


## <span id="Acknowledgements"> Acknowledgements
This implementation is based on code from several repositories.
- [Carla leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario runner](https://github.com/carla-simulator/scenario_runner)
- [Interfuser](https://github.com/opendilab/InterFuser)
- [Opencood](https://github.com/DerrickXuNu/OpenCOOD)
- [HEAL](https://github.com/yifanlu0227/HEAL)

## Citation
```
@article{liu2024codriving,
  title={Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System},
  author={Liu, Genjia and Hu, Yue and Xu, Chenxin and Mao, Weibo and Ge, Junhao and Huang, Zhengxiang and Lu, Yifan and Xu, Yinda and Xia, Junkai and Wang, Yafei and others},
  journal={arXiv preprint arXiv:2404.09496},
  year={2024}
}
```
