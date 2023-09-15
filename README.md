# V2Xverse: a codebase for V2X based end2end autonomous driving

## Contents
1. [Installation](#Installation)
2. [Data Generation](#Data-generation)
3. [Training](#Training)
4. [Visualization](#Visualization)
5. [Git 开发](#Git开发)
6. [Acknowledgements](#Acknowledgements)

## Installation

### Carla 所需依赖

参考carla leaderboard的[配置步骤](https://leaderboard.carla.org/get_started/)

```Shell
# Carla 环境配置步骤

TODO
```
### Download Carla
Download and setup CARLA 0.9.10.1
```Shell
# 下载carla压缩包，解压，删除压缩包
cd simulation
chmod +x setup_carla.sh
./setup_carla.sh
easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```

### Opencoodv2所需依赖

参考 [飞书文档](https://udtkdfu8mk.feishu.cn/docx/doxcnMGYWWnJq3qIR9obqPRbmoZ)

## Data preparation

V2Xverse dataset can be downloaded from link(TODO)

Make sure the dataset follow this structure
```Shell
# TODO: dataset file structure



```

### Data content illustration

A doc for data annotation (eg. [飞书文档](https://udtkdfu8mk.feishu.cn/mindnotes/bmncnNo9xElMoWMidFZxpmF08fg))

### Generate data by yourself

Users can also generate data by running simulation process on carla.


~~~Shell
# Generate the whole dataset in parallel

# Initialize dataset directory
bash scripts/init_dataset_dir.sh [your dataset directory]

# Generate config files (weather 0-13)
python simulation/data_collection/generate_yamls.py

# Generate scripts for every routes
python simulation/data_collection/generate_scripts.py

# Packing the scripts
python generate_batch_collect.py

# Open Carla server （15 parallel process in total）
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40000 -prefer-nvidia
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40002 -prefer-nvidia
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40004 -prefer-nvidia
...
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40028 -prefer-nvidia

# Execute data generation in parallel
bash simulation/data_collection/generate_v2xverse_all.sh
~~~

Command to generate data for one single route
~~~Shell
# Open one Carla server
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40000 -prefer-nvidia

# Execute data generation for route 0 in town01
bash simulation/data_collection/routes_town01_0.sh
~~~


## Planner training

To train the planner, link the prepared dataset to a specific path under the project folder, then run the planner training script

```BASH
ln -s ${YOUR_DATA_LOCATION} ./data
bash scripts/train_planner.sh
```


### Linux进程的处理
carla可能会存在结束进程失败的情况，及时处理垃圾进程

[参考链接](https://blog.csdn.net/u010227042/article/details/127158397)

显示自己的进程
~~~
ps U usrname | grep str(要查询的进程字段，例如python，carla)
~~~
强制杀死进程
~~~
kill -9 PID
~~~
批量删除有关carla的进程，这条命令会终止所有相关进程，但由于普通用户的权限不够不会影响其他用户，root用户勿用
~~~
ps -def |grep 'carla' |cut -c 9-15| xargs kill -9
~~~

## Close-loop evaluation

```Shell
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=3000 -prefer-nvidia
bash scripts/eval_pnp.sh
```

## Visualization
```Shell

TODO
```


## Acknowledgements
This implementation is based on code from several repositories.
- [Interfuser](https://github.com/opendilab/InterFuser)
- [Carla leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario runner](https://github.com/carla-simulator/scenario_runner)
