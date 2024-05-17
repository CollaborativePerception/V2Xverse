import numpy as np
from collections import deque
from team_code.render_v2x import render, render_self_car

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

def downsample_waypoints(waypoints, precision=0.2):
    """
    waypoints: [float lits], 10 * 2, m
    """
    downsampled_waypoints = []
    downsampled_waypoints.append(np.array([0, 0]))
    last_waypoint = np.array([0.0, 0.0])
    for i in range(10):
        now_waypoint = waypoints[i]
        dis = np.linalg.norm(now_waypoint - last_waypoint)
        if dis > precision:
            interval = int(dis / precision)
            move_vector = (now_waypoint - last_waypoint) / (interval + 1)
            for j in range(interval):
                downsampled_waypoints.append(last_waypoint + move_vector * (j + 1))
        downsampled_waypoints.append(now_waypoint)
        last_waypoint = now_waypoint
    return downsampled_waypoints

def collision_detections(map1, map2, threshold=0.04):
    """
    map1: rendered surround vehicles
    map2: self-car
    """
    assert map1.shape == map2.shape
    overlap_map = (map1 > 0.01) & (map2 > 0.01)
    ratio = float(np.sum(overlap_map)) / np.sum(map2 > 0)
    ratio2 = float(np.sum(overlap_map)) / np.sum(map1 > 0)
    if ratio < threshold:
        return True
    else:
        return False

def get_max_safe_distance(meta_data, downsampled_waypoints, t, collision_buffer, threshold):
    surround_map = render(meta_data.reshape(20, 20, 7), t=t)[0][:100, 40:140]
    if np.sum(surround_map) < 1:
        return np.linalg.norm(downsampled_waypoints[-3])
    # need to render self-car map
    hero_bounding_box = np.array([2.45, 1.0]) + collision_buffer
    safe_distance = 0.0
    for i in range(len(downsampled_waypoints) - 2):
        aim = (downsampled_waypoints[i + 1] + downsampled_waypoints[i + 2]) / 2.0
        loc = downsampled_waypoints[i]
        ori = aim - loc
        self_car_map = render_self_car(loc=loc, ori=ori, box=hero_bounding_box)[
            :100, 40:140
        ]
        if collision_detections(surround_map, self_car_map, threshold) is False:
            break
        safe_distance = max(safe_distance, np.linalg.norm(loc))
    return safe_distance

class V2X_Controller(object):
    def __init__(self, config):
        self.turn_controller = PIDController(
            K_P=config.turn_KP, 
            K_I=config.turn_KI, 
            K_D=config.turn_KD, 
            n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP,
            K_I=config.speed_KI,
            K_D=config.speed_KD,
            n=config.speed_n,
        )
        self.collision_buffer = np.array(config.collision_buffer)
        self.config = config
        self.detect_threshold = config.detect_threshold
        self.stop_steps = 0
        self.forced_forward_steps = 0

        self.red_light_steps = 0
        self.block_red_light = 0

        self.block_stop_sign_distance = (
            0  # it means in 30m, stop sign will not take effect again
        )
        self.stop_sign_trigger_times = 0

    def run_step(
        self, route_info
    ):
        """
        Currently, we generate the desired speed according to predicted waypoints only!
        In the next step, we need to consider the GLOBAL speed to finish the route in time.

        route_info: {
            'speed': float, m/s, current speed,
            'waypoints': [float list], 10 * 2, m,
            'target':
            'route_length': m,
            'route_time': s,
            'drive_length': m,
            'drive_time': s
        }
        """
        speed = route_info['speed']
        waypoints = np.array(route_info['waypoints'])
        if speed < 0.2:
            self.stop_steps += 1
        else:
            self.stop_steps = max(0, self.stop_steps - 10)

        aim = route_info['target']
        aim_wp = (waypoints[-2] + waypoints[-1]) / 2.0
        theta_tg = np.arctan2(aim[0], aim[1]+0.0000001)
        
        angle_tg = np.sign(theta_tg) * (180 - np.abs(np.degrees(theta_tg))) / 90
        angle_wp = np.degrees(np.pi / 2 - np.arctan2(aim_wp[1], aim_wp[0]+0.0000005)) / 90

        weight = 1
        angle = angle_wp * (1-weight) + angle_tg * weight
        if speed < 0.01:
            angle = 0
        # angle = np.around(angle, decimals=2)
        # print(waypoints, aim, theta_, angle)
        steer = self.turn_controller.step(angle)
        # steer = np.around(steer, decimals=2)
        steer = np.clip(steer, -1.0, 1.0)

        brake = False
        # get desired speed according to the future waypoints
        displacement = np.linalg.norm(waypoints, ord=2, axis=1)
        desired_speed = np.mean(np.diff(displacement)[:3]) * 5 * self.config.max_speed / 5
        # v 
        # desired_speed = np.around(desired_speed, decimals=2)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        # throttle = np.around(desired_speed, decimals=2)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)

        if speed > desired_speed * self.config.brake_ratio:
            brake = True

        # stop for too long, force to forward
        if self.stop_steps > 100:
            self.forced_forward_steps = 12
            self.stop_steps = 0
        if self.forced_forward_steps > 0:
            throttle = 0.8
            brake = False
            self.forced_forward_steps -= 1


        meta_info_1 = "speed: %.2f, target_speed: %.2f, angle: %.2f, [%.2f, %.2f, %.2f, %.2f, %.2f]" % (
            speed,
            desired_speed, angle,
            self.turn_controller._window[0], self.turn_controller._window[1], self.turn_controller._window[2], self.turn_controller._window[3], self.turn_controller._window[4]
        )
        meta_info_2 = "stop_steps:%d" % (
            self.stop_steps
        )
        meta_info = {
            1: meta_info_1,
            2: meta_info_2,
        }


        return steer, throttle, brake, meta_info
