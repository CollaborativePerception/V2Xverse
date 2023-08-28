import yaml
import os

d = {}
d[
    "waypoint_disturb"
] = 0.2  # in meters, a way of data augumentaion that randomly distrub the planned waypoints
d["waypoint_disturb_seed"] = 2020
d["destroy_hazard_actors"] = True
d["save_skip_frames"] = 4  # skip 4 frames equals fps = 5
d["rgb_only"] = False
d["target_point_distance"] = 50
d["max_speed"] = 6.5

if not os.path.exists("yamls"):
    os.mkdir("yamls")

for i in range(14):
    d["weather"] = i
    with open("yamls/weather-%d.yaml" % i, "w") as fw:
        yaml.dump(d, fw)
