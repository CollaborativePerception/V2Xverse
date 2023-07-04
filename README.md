

## Data generation

先开carla server, 注意调整port和traffic manager port

~~~
CUDA_VISIBLE_DEVICES=0 /GPFS/data/gjliu/Auto-driving/Cop3/carla/CarlaUE4.sh --world-port=60000 -prefer-nvidia
~~~

运行数据生成
~~~
bash simulation/data_collection/routes_town01_0.sh
~~~