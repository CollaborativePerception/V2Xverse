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
# Initialize dataset directory
python simulation/data_collection/init_dir.py --dataset_dir [your dirctory]

# Generate config files (weather 0-13)
cd simulation/data_collection
python generate_yamls.py

# Generate shells for data collection
python generate_bashs.py

~~~
### 路径初始化文件的地址
存储在 leaderboard/data/training_routes/cop3_split_routes下
### 计算消耗
单route大概2GB显存
### 单条route数据生成
先开carla server, 注意调整port和traffic manager port, 

~~~
CUDA_VISIBLE_DEVICES=0 /GPFS/data/gjliu/Auto-driving/Cop3/carla/CarlaUE4.sh(or your carla path) --world-port=40000 -prefer-nvidia
~~~

运行数据生成
~~~
bash simulation/data_collection/routes_town01_0.sh
~~~


## Planner training

To train the planner, link the prepared dataset to a specific path under the project folder, then run the planner training script

```BASH
ln -s ${YOUR_DATA_LOCATION} ./data
bash scripts/train_planner.sh
```

## data generation
bash data_collection/weather-0/routes_town01_0.sh
~~~
### 批量生成命令

先开carla server，注意调整CUDA device和world-port
~~~
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40000 -prefer-nvidia
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40002 -prefer-nvidia
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40004 -prefer-nvidia
...
CUDA_VISIBLE_DEVICES=0 carla/CarlaUE4.sh --world-port=40030 -prefer-nvidia
~~~

根据打开的端口（world-port），生成运行指令
~~~
python generate_bashs.py
~~~
打包指令
~~~
python generate_batch_collect.py
~~~
运行数据生成simulation
~~~
bash data_collection/generate_cop3_all.sh
~~~


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


## Training
```Shell
cd interfuser
bash scripts/train.sh
bash scripts/mao_train.sh
bash scripts/planner_train.sh
## early fusion
FUSE_MODE='early' bash scripts/dev_cop3_train.sh

## inter fusion(where2comm)
FUSE_MODE='inter' bash scripts/dev_cop3_train.sh
```
分布式训练可能存在未能完全关闭进程的情况，请及时清理

## Evaluation

CUDA_VISIBLE_DEVICES=3 ./CarlaUE4.sh --world-port=3000 -opengl


CUDA_VISIBLE_DEVICES=6 ./leaderboard/scripts/eval_cop3_short_cheat.sh
CUDA_VISIBLE_DEVICES=2 ./leaderboard/scripts/eval_cop3_short_none.sh



## Visualization
```Shell
cd visualization
python check_3d_box_with_lidar.py
python check_3d_bbs.py
python check_3d_bbs_early_fusion.py
```

## Git开发
初始化
Git global setup
```Shell
git config --global user.name your_username
git config --global user.email your_email
```

```Shell
# 克隆远程仓库的 dev 分支到本地，若已完成则跳过这一步
git clone -b dev http://202.120.39.225:58280/LGJ1zed/CoP3.git

# 切换到本地的dev_loc分支（如果没有则新建一个），每次开发前先获取远程 dev 分支最新代码
git checkout dev_loc
git pull origin dev:dev_loc

# 提交代码
git add . # 注意，这里的 “.” 代表文件名，这里是指添加工作区所有文件到暂存区，请将不想添加的文件写入.gitignore，务必避免添加大型数据文件 
git commit -m "your message"
# 提交到远程
git push origin dev_loc:dev

# 避免重复输入账号密码
git config --global credential.helper store

# 强制pull覆盖本地
# 运行 fetch 以将所有 origin/ 引用更新为最新
git fetch --all
# 可以备份当前分支
git branch backup-master
# 强制拉取origin/master
git reset --hard origin/master
```

## Acknowledgements
This implementation is based on code from several repositories.
- [Interfuser](https://github.com/opendilab/InterFuser)
- [Carla leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario runner](https://github.com/carla-simulator/scenario_runner)
