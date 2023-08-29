import yaml
import os

d = {}

# driving control params
d["waypoint_disturb"] = 0.2             # in meters, a way of data augumentaion that randomly distrub the planned waypoints
d["waypoint_disturb_seed"] = 2020       #
d["destroy_hazard_actors"] = True       # 'True' is recommended in crazy scenario, to destroy
d["target_point_distance"] = 50         # distance between two adjacent target waypoint
d["max_speed"] = 6.5                    # m/s, max speed of ego vehicle

# data collection params
d["save_skip_frames"] = 4               # the number of simulation frames between two data saves (default simulation frequency is 20hz)
d["rgb_only"] = False                   # flag to only save camera sensor data

# road side unit (rsu) params
d["use_rsu"] = True                     # whether to collect data from road side unit
d["change_rsu_frame"] = 100             # the number of simulation frames between two times of accessing to a new rsu
d["rsu_height"] = 7.5                   # m, height of rsu
d["rsu_lane_side"] = 'right'            # 'right'/'left', which lane side to spawn rsu
d["rsu_distance"] = 12                  # m, distance from ego vechicle to the a accessed rsu

if not os.path.exists("simulation/data_collection/yamls"):
    os.mkdir("simulation/data_collection/yamls")

for i in range(14):
    d["weather"] = i
    with open("simulation/data_collection/yamls/weather-%d.yaml" % i, "w") as fw:
        yaml.dump(d, fw)
