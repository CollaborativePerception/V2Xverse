

## Data generation

先开carla server, 注意调整port和traffic manager port

~~~
CUDA_VISIBLE_DEVICES=0 /GPFS/data/gjliu/Auto-driving/Cop3/carla/CarlaUE4.sh --world-port=60000 -prefer-nvidia
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
