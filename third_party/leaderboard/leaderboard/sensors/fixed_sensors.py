import pathlib
import carla
import json
import cv2
import os
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.envs.sensor_interface import CallBack, OpenDriveMapReader, SpeedometerReader, SensorConfigurationInvalid
from leaderboard.autoagents.autonomous_agent import Track
from leaderboard.envs.sensor_interface import SensorInterface
from srunner.scenariomanager.timer import GameTime
from srunner.tools.scenario_helper import get_location_in_distance_from_wp, get_location_in_distance_from_wp_left
from PIL import Image, ImageDraw
import numpy as np
from team_code.eval_utils import get_yaw_angle, boxes_to_corners_3d, get_points_in_rotated_box_3d, process_lidar_visibility

from team_code.utils.map_drawing import \
    cv2_subpixel, draw_agent, draw_road, \
    draw_lane, road_exclude, draw_crosswalks, draw_city_objects
from team_code.utils.map_utils import \
    world_to_sensor, lateral_shift, list_loc2array, list_wpt2array, \
    convert_tl_status, exclude_off_road_agents, retrieve_city_object_info, \
    obj_in_range

TPE = {
    carla.CityObjectLabel.Buildings: "Building", 
    carla.CityObjectLabel.Vegetation: "Vegetation", 
    carla.CityObjectLabel.Poles: "Poles",
    carla.CityObjectLabel.Pedestrians: "Pedestrian",
    carla.CityObjectLabel.TrafficSigns: "TrafficSign",
    carla.CityObjectLabel.TrafficLight: "TrafficLight",
    carla.CityObjectLabel.Static: "Static",
    carla.CityObjectLabel.Fences: "Fence",
    carla.CityObjectLabel.Walls: "Wall"
}


def get_rsu_point(vehicle, height=7.5, lane_side='right', distance=12):
    """
    Args:
        vehicle: (carla.Actor) the vehicle that rsu corresponds to.
        height: (int) the height of the rsu.
        lane_side: spawn rsu on right/left side of lane
        distance: (float) the distance from RSU to the car, positive value means rsu is in front of car
    Returns:
        (carla.Loction): The exact suggested spawn loction for the rsu. 
    """
    vehicle_loc = vehicle.get_location()
    vehicle_wp = CarlaDataProvider.get_map().get_waypoint(vehicle_loc)
    spawn_point = vehicle_wp
    while vehicle_wp.get_right_lane():
        spawn_point = vehicle_wp.get_right_lane() 
        vehicle_wp = vehicle_wp.get_right_lane() 
    # spawn_loc = spawn_point.transform.location

    if distance >= 0:
        direction = 'foward'
    else:
        direction = 'backward'

    #### adjust the RSU distance.
    if lane_side == 'right':
        spawn_loc, _ = get_location_in_distance_from_wp(spawn_point, distance, False, direction)
    else:
        spawn_loc, _ = get_location_in_distance_from_wp_left(spawn_point, distance, False, direction)
    return carla.Location(
        x=spawn_loc.x,
        y=spawn_loc.y,
        z=spawn_loc.z+height
    )

def get_camera_intrinsic(sensor):
    """
    Retrieve the camera intrinsic matrix.
    Parameters
    ----------
    sensor : carla.sensor
        Carla rgb camera object.
    Returns
    -------
    matrix_x : list
        The 2d intrinsic matrix.
    """
    VIEW_WIDTH = int(sensor['width'])
    VIEW_HEIGHT = int(sensor['height'])
    VIEW_FOV = int(float(sensor['fov']))
    # VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    # VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    # VIEW_FOV = int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    return matrix_k.tolist()

def get_camera_extrinsic(cur_extrin, ref_extrin):
    """
    Args:
        cur_extrin (carla.Transform): current extrinsic
        ref_extrin (carla.Transform): reference extrinsic
    Returns:
        extrin (list): 4x4 extrinsic matrix with respect to reference coordinate system 
    """
    extrin = np.array(ref_extrin.get_inverse_matrix()) @ np.array(cur_extrin.get_matrix())
    return extrin.tolist()

def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

def find_obstacle_3dbb(ego_loc, obstacle_type, max_distance=50):
    """Returns a list of 3d bounding boxes of type obstacle_type.
    If the object does have a bounding box, this is returned. Otherwise a bb
    of size 0.5,0.5,2 is returned at the origin of the object.

    Args:
        ego_loc (carla.Location): can be passed to get the object by name from carla world 
        obstacle_type (String): Regular expression
        max_distance (int, optional): max search distance. Returns all bbs in this radius. Defaults to 50.

    Returns:
        List: List of Boundingboxes
    """
    obst = list()
    # TODO: Check it! Could conceal hidden
    if obstacle_type == "*traffic_light*" or obstacle_type == "*stop*":
        # Retrieve all bounding boxes for traffic lights within the level
        if obstacle_type == "*traffic_light*":
            bounding_box_set = CarlaDataProvider.get_world().get_level_bbs(carla.CityObjectLabel.TrafficLight)
        else:
            bounding_box_set = CarlaDataProvider.get_world().get_level_bbs(carla.CityObjectLabel.TrafficSigns)
        # Filter the list to extract bounding boxes within a 50m radius
        for bbox in bounding_box_set:
            distance_to_car = bbox.location.distance(ego_loc)
            if 0 < distance_to_car <= max_distance:
                loc = bbox.location
                ori = bbox.rotation
                extent = bbox.extent
                bb = np.array(
                    [
                        [loc.x, loc.y, loc.z],
                        [ori.roll, ori.pitch, ori.yaw],
                        [extent.x, extent.y, extent.z]
                        # [rotated_extent[0], rotated_extent[1], rotated_extent[2]],
                    ]
                )

                obst.append(bb)
    else: # original
        _actors = CarlaDataProvider.get_world().get_actors()
        _obstacles = _actors.filter(obstacle_type)

        for _obstacle in _obstacles:
            distance_to_car = _obstacle.get_transform().location.distance(ego_loc)

            if 0 < distance_to_car <= max_distance:
                if hasattr(_obstacle, "bounding_box"):
                    loc = _obstacle.get_location()
                    ori = _obstacle.get_transform().rotation.get_forward_vector()
                    extent = _obstacle.bounding_box.extent
                    bb = np.array(
                        [
                            [loc.x, loc.y, loc.z],
                            [ori.x, ori.y, ori.z],
                            [extent.x, extent.y, extent.z]
                            # [rotated_extent[0], rotated_extent[1], rotated_extent[2]],
                        ]
                    )

                else:
                    loc = _obstacle.get_transform().location
                    bb = np.array([[loc.x, loc.y, loc.z], [0.5, 0.5, 2]])

                obst.append(bb)

    return obst
        
class SensorUnit(object):
    def __init__(self, save_path, id=0, is_None=False):
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        self._sensor_data = self._rgb_sensor_data.copy()
        self._parent_carla_id = None
        self.name = "rsu"
        self.deleted=False
        self._sensors_list = []

    def pose_def(self):
        cam_pitch = -40. # the angle looking down.
        left_yaw = -90. # originally -60.0
        right_yaw = 90. # originally 60.0
        self.lidar_pose = carla.Transform(carla.Location(x=0.0,y=0.0,z=0.0),
                                        carla.Rotation(roll=0.0,pitch=0.0,yaw=-90.0))
        self.camera_front_pose = carla.Transform(carla.Location(x=0.0,y=0.0,z=0.0),
                                        carla.Rotation(roll=0.0,pitch=cam_pitch,yaw=0.0))
        self.camera_rear_pose = carla.Transform(carla.Location(x=0.0,y=0.0,z=0.0),
                                        carla.Rotation(roll=0.0,pitch=cam_pitch,yaw=180.0))
        self.camera_left_pose = carla.Transform(carla.Location(x=0.0,y=0.0,z=0.0),
                                        carla.Rotation(roll=0.0,pitch=cam_pitch,yaw=left_yaw))
        self.camera_right_pose = carla.Transform(carla.Location(x=0.0,y=0.0,z=0.0),
                                        carla.Rotation(roll=0.0,pitch=cam_pitch,yaw=right_yaw))
        return

    def sensors(self, results=None):
        if not isinstance(results,list):
            results=[]
        self.pose_def()
        config_list = [
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_front",
            },
            {
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_front",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_front",
            },
            {
                "type": "sensor.lidar.ray_cast_semantic",
                "x": self.camera_front_pose.location.x,
                "y": self.camera_front_pose.location.y,
                "z": self.camera_front_pose.location.z,
                "roll": self.camera_front_pose.rotation.roll,
                "pitch": 0, # self.camera_front_pose.rotation.pitch,
                "yaw": self.camera_front_pose.rotation.yaw,
                "id": "lidar_semantic_front",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_rear",
            },
            {
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_rear",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_rear_pose.location.x,
                "y": self.camera_rear_pose.location.y,
                "z": self.camera_rear_pose.location.z,
                "roll": self.camera_rear_pose.rotation.roll,
                "pitch": self.camera_rear_pose.rotation.pitch,
                "yaw": self.camera_rear_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_rear",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_left",
            },
            {
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_left",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_left_pose.location.x,
                "y": self.camera_left_pose.location.y,
                "z": self.camera_left_pose.location.z,
                "roll": self.camera_left_pose.rotation.roll,
                "pitch": self.camera_left_pose.rotation.pitch,
                "yaw": self.camera_left_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_left",
            },
            {
                "type": "sensor.camera.rgb",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._rgb_sensor_data["width"],
                "height": self._rgb_sensor_data["height"],
                "fov": self._rgb_sensor_data["fov"],
                "id": "rgb_right",
            },
            {
                "type": "sensor.camera.semantic_segmentation",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "seg_right",
            },
            {
                "type": "sensor.camera.depth",
                "x": self.camera_right_pose.location.x,
                "y": self.camera_right_pose.location.y,
                "z": self.camera_right_pose.location.z,
                "roll": self.camera_right_pose.rotation.roll,
                "pitch": self.camera_right_pose.rotation.pitch,
                "yaw": self.camera_right_pose.rotation.yaw,
                "width": self._sensor_data["width"],
                "height": self._sensor_data["height"],
                "fov": self._sensor_data["fov"],
                "id": "depth_right",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": self.lidar_pose.location.x,
                "y": self.lidar_pose.location.y,
                "z": self.lidar_pose.location.z,
                "roll": self.lidar_pose.rotation.roll,
                "pitch": self.lidar_pose.rotation.pitch,
                "yaw": self.lidar_pose.rotation.yaw,
                "id": "lidar",
            }
        ]
        results.extend(config_list)
        return results

    def setup_sensors(self, parent=None, spawn_point=None):
        """
        Create the sensors.
        Args:
        Returns:
        """
        # decide the spawn locations
        self._rsu_loc = carla.Location()
        if parent is not None:
            if isinstance(parent,carla.Actor):
                self._rsu_loc = parent.get_location()
        elif spawn_point is not None:
            self._rsu_loc = spawn_point
        # print([self._rsu_loc.x, self._rsu_loc.y, self._rsu_loc.z])
        # spawn the sensors
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        for sensor_spec in self.sensors():
            # These are the pseudosensors (not spawned)
            bp = bp_library.find(str(sensor_spec['type']))
            if sensor_spec['type'].startswith('sensor.camera.semantic_segmentation'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))

                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.camera.depth'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))

                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(sensor_spec['width']))
                bp.set_attribute('image_size_y', str(sensor_spec['height']))
                bp.set_attribute('fov', str(sensor_spec['fov']))
                bp.set_attribute('lens_circle_multiplier', str(1.0))
                bp.set_attribute('lens_circle_falloff', str(1.0))
                bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                bp.set_attribute('chromatic_aberration_offset', str(0))

                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.lidar.ray_cast_semantic'):
                bp.set_attribute('range', str(85))
                bp.set_attribute('rotation_frequency', str(10)) # default: 10, change to 20 for old lidar models
                bp.set_attribute('channels', str(64))
                bp.set_attribute('upper_fov', str(10))
                bp.set_attribute('lower_fov', str(-60))
                bp.set_attribute('points_per_second', str(600000))
                sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(20)) # default: 10, change to 20 to generate 360 degree LiDAR point cloud
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-60))
                    bp.set_attribute('points_per_second', str(1000000))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    #### NOTE: changed, 0.45 -> 0.0
                    bp.set_attribute('dropoff_general_rate', str(0))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(x=sensor_spec['x'], y=sensor_spec['y'],
                                                        z=sensor_spec['z'])
                    sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                        roll=sensor_spec['roll'],
                                                        yaw=sensor_spec['yaw'])
            elif sensor_spec['type'].startswith('sensor.other.radar'):
                bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                bp.set_attribute('points_per_second', '1500')
                bp.set_attribute('range', '100')  # meters

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                    y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])

            elif sensor_spec['type'].startswith('sensor.other.gnss'):
                # bp.set_attribute('noise_alt_stddev', str(0.000005))
                # bp.set_attribute('noise_lat_stddev', str(0.000005))
                # bp.set_attribute('noise_lon_stddev', str(0.000005))
                bp.set_attribute('noise_alt_bias', str(0.0))
                bp.set_attribute('noise_lat_bias', str(0.0))
                bp.set_attribute('noise_lon_bias', str(0.0))

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                    y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation()

            elif sensor_spec['type'].startswith('sensor.other.imu'):
                bp.set_attribute('noise_accel_stddev_x', str(0.001))
                bp.set_attribute('noise_accel_stddev_y', str(0.001))
                bp.set_attribute('noise_accel_stddev_z', str(0.015))
                bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                bp.set_attribute('noise_gyro_stddev_z', str(0.001))

                sensor_location = carla.Location(x=sensor_spec['x'],
                                                    y=sensor_spec['y'],
                                                    z=sensor_spec['z'])
                sensor_rotation = carla.Rotation(pitch=sensor_spec['pitch'],
                                                    roll=sensor_spec['roll'],
                                                    yaw=sensor_spec['yaw'])
            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)

            # Check if attachment is needed.
            if parent is not None:
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, attach_to=parent)
            elif spawn_point is not None:
                sensor_transform = carla.Transform(
                    carla.Location(
                        sensor_transform.location.x+spawn_point.x,
                        sensor_transform.location.y+spawn_point.y,
                        sensor_transform.location.z+spawn_point.z
                    ),
                    sensor_transform.rotation
                )
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform)
            else:
                sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform)

            # setup callback
            sensor.listen(CallBack(self.name+"_{}_".format(self.id)+sensor_spec['id'], sensor_spec['type'], sensor, self.sensor_interface))
            
            self._sensors_list.append(sensor)

        # Tick once to spawn the sensors
        # Tick must be suspended.
        # CarlaDataProvider.get_world().tick()

    def _get_3d_bbs(self, max_distance=50):
        bounding_boxes = {
            "traffic_lights": [],
            "stop_signs": [],
            "vehicles": [],
            "pedestrians": [],
        }

        bounding_boxes["traffic_lights"] = find_obstacle_3dbb(self._rsu_loc, "*traffic_light*", max_distance)
        bounding_boxes["stop_signs"] = find_obstacle_3dbb(self._rsu_loc, "*stop*", max_distance)
        bounding_boxes["vehicles"] = find_obstacle_3dbb(self._rsu_loc, "*vehicle*", max_distance)
        bounding_boxes["pedestrians"] = find_obstacle_3dbb(self._rsu_loc, "*walker.*", max_distance)

        return bounding_boxes

    def run_step(self, frame):
        """
        Always, you need to rewrite this function to enable your sensor units.
        """

    def tick(self):
        if not self.deleted:
            return self.sensor_interface.get_data()
        return None

    def collect_actor_data(self):
        data = {}
        vehicles = CarlaDataProvider.get_world().get_actors().filter("*vehicle*")
        for actor in vehicles:
            loc = actor.get_location()
            if loc.distance(self._rsu_loc) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y, box.z]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            angle = actor.get_transform().rotation
            data[_id]["angle"] = [angle.roll, angle.yaw, angle.pitch]
            if actor.type_id=="vehicle.diamondback.century":
                data[_id]["tpe"] = 3
            else:
                data[_id]["tpe"] = 0

        walkers = CarlaDataProvider.get_world().get_actors().filter("*walker.*")
        for actor in walkers:
            loc = actor.get_location()
            if loc.distance(self._rsu_loc) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            box = actor.bounding_box.extent
            data[_id]["box"] = [box.x, box.y, box.z]
            vel = actor.get_velocity()
            data[_id]["vel"] = [vel.x, vel.y, vel.z]
            angle = actor.get_transform().rotation
            data[_id]["angle"] = [angle.roll, angle.yaw, angle.pitch]
            data[_id]["tpe"] = 1

        lights = CarlaDataProvider.get_world().get_actors().filter("*traffic_light*")
        for actor in lights:
            loc = actor.get_location()
            if loc.distance(self._rsu_loc) > 50:
                continue
            _id = actor.id
            data[_id] = {}
            data[_id]["loc"] = [loc.x, loc.y, loc.z]
            ori = actor.get_transform().rotation.get_forward_vector()
            data[_id]["ori"] = [ori.x, ori.y, ori.z]
            vel = actor.get_velocity()
            data[_id]["sta"] = int(actor.state)
            data[_id]["tpe"] = 2

            trigger = actor.trigger_volume
            box = trigger.extent
            loc = trigger.location
            ori = trigger.rotation.get_forward_vector()
            data[_id]["taigger_loc"] = [loc.x, loc.y, loc.z]
            data[_id]["trigger_ori"] = [ori.x, ori.y, ori.z]
            data[_id]["trigger_box"] = [box.x, box.y, box.z]
        return data

    def collect_env_actor_data(self):
        data = {}
        _id = 0
        for actor_type in list(TPE.keys()):
            actors = CarlaDataProvider.get_world().get_level_bbs(actor_type=actor_type)
            for actor in actors:
                loc = actor.location
                if loc.distance(self._rsu_loc) > 50:
                    continue
                _id += 1
                data[_id] = {}
                data[_id]["loc"] = [loc.x, loc.y, loc.z]
                ori = actor.rotation.get_forward_vector()
                data[_id]["ori"] = [ori.x, ori.y, ori.z]
                box = actor.extent
                data[_id]["box"] = [box.x, box.y, box.z]
                data[_id]["tpe"] = TPE[actor_type]

        
        return data


    def del_sensors(self):
        """
        Remove sensor tags of one rsu
        """
        for sensor_spec in self.sensors():
            del self.sensor_interface._sensors_objects[self.name+'_'+sensor_spec['id']+"_{}".format(self.id)]
        self.deleted=True

    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for _sensor in self._sensors_list:
            if _sensor is not None:
                _sensor.stop()
                _sensor.destroy()
                _sensor = None
        self._sensors_list = []
        self.deleted=True

# instantiated sensor units

class TrafficLightSensor(SensorUnit):
    def __init__(self, save_path, id=0, is_None=False):
        super().__init__()        
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        self.save_path = save_path
        self.id = id
        self.save_path_tmp =self.save_path / pathlib.Path("traffic_light_{}".format(self.id))
        self.loc_path = self.save_path_tmp / "location.json"
        # (self.save_path_tmp / "lidar").mkdir(parents=True, exist_ok=True)
        for pos in ["front", "rear"]:
            for sensor_type in ["rgb", "lidar"]:
                name = sensor_type + "_" + pos
                (self.save_path_tmp / name).mkdir(parents=True, exist_ok=True)
        (self.save_path_tmp / "3d_bbs").mkdir(parents=True, exist_ok=True)
        (self.save_path_tmp / "status").mkdir(parents=True, exist_ok=True)
        self.sensor_interface = SensorInterface()
        self._3d_bb_distance = 85
        self.name = "tf"

    def run_step(self, frame):
        """
        """
        if frame is not None:
            # read the data from sensor
            input_data = self.sensor_interface.get_data()
            # preprocess the data, temporarily store it in tick_data
            rgb_front = cv2.cvtColor(input_data["tf_{}_rgb_front".format(self.id)][1][:, :, :3], cv2.COLOR_BGR2RGB)
            rgb_rear = cv2.cvtColor(input_data["tf_{}_rgb_rear".format(self.id)][1][:, :, :3], cv2.COLOR_BGR2RGB)
            lidar_front = input_data["tf_{}_lidar_front".format(self.id)][1]
            lidar_rear = input_data["tf_{}_lidar_rear".format(self.id)][1]
            tick_data = {
                "tf_{}_rgb_front".format(self.id): rgb_front,
                "tf_{}_rgb_rear".format(self.id): rgb_rear,
                "tf_{}_lidar_front".format(self.id) : lidar_front,
                "tf_{}_lidar_rear".format(self.id) : lidar_rear
            }
            # store the data
            for pos in ["front", "rear"]:
                name = "tf_{}_rgb_".format(self.id) + pos              
                name_save = "rgb_" + pos
                Image.fromarray(tick_data[name]).save(
                    self.save_path_tmp / name_save / ("%04d.jpg" % frame)
                )
                np.save(
                    self.save_path_tmp / ("lidar_"+pos) / ("%04d.npy" % frame),
                    tick_data["tf_{}_lidar_".format(self.id) + pos],
                    allow_pickle=True,
                )
            bb_3d = self._get_3d_bbs(max_distance=self._3d_bb_distance)
            np.save(
                self.save_path_tmp / "3d_bbs" / ("%04d.npy" % frame),
                bb_3d,
                allow_pickle=True,
            )

            # store the ego traffic light state
            self._tf = CarlaDataProvider.get_world().get_actors().find(self._tf_carla_id)
            if self._tf.get_state() == carla.TrafficLightState.Red:
                color = "Red"
                seg_state = 23
            elif self._tf.get_state() == carla.TrafficLightState.Yellow:
                color = "Yellow"
                seg_state = 24
            elif self._tf.get_state() == carla.TrafficLightState.Green:
                color = "Green"
                seg_state = 25
            else:  # none of the states above, do not change class
                color = "Other"
                seg_state = 18
            with open((self.save_path_tmp / "status" / ("%04d.json" % frame)), "wt") as f:
                json.dump({"state": color, "seg_mask": seg_state},f,indent=4,separators=[",",": "])
        else:
            return


class RoadSideUnit(SensorUnit):
    def __init__(self, save_path, id=0, is_None=False):
        super().__init__(save_path,id)        
        if is_None:
            return
        self._rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
        self.save_path = save_path
        self.id = id
        if not self.save_path is None:
            self.save_path_tmp =self.save_path / pathlib.Path("rsu_{}".format(self.id))
            self.loc_path = self.save_path_tmp / "location.json"
            # intialize folders for the sensors' data.
            (self.save_path_tmp / "lidar").mkdir(parents=True, exist_ok=True)
            (self.save_path_tmp / "lidar_semantic_front").mkdir(parents=True, exist_ok=True)
            for sensor_type in ["rgb", "depth"]:
                for pos in ["front", "left", "right", "rear"]:
                    name = sensor_type + "_" + pos
                    (self.save_path_tmp / name).mkdir(parents=True, exist_ok=True)
            (self.save_path_tmp / "3d_bbs").mkdir(parents=True, exist_ok=True)
            (self.save_path_tmp / "measurements").mkdir(parents=True, exist_ok=True)
            (self.save_path_tmp / "actors_data").mkdir(parents=True, exist_ok=True)
            (self.save_path_tmp / "env_actors_data").mkdir(parents=True, exist_ok=True)
            (self.save_path_tmp / "bev_visibility").mkdir(parents=True, exist_ok=True)
        self.sensor_interface = SensorInterface()
        self._3d_bb_distance = 85

    def process(self,input_data,is_train=False):
        rgb_data = {}
        results = {}
        try:
            for pos in ["front", "left", "right", "rear"]:
                rgb_data[pos] = cv2.cvtColor(
                    input_data["rsu_{0}_rgb_{1}".format(self.id,pos)][1][:, :, :3],
                    cv2.COLOR_BGR2RGB
                )
                name_save = "rgb_" + pos
                results[name_save]=rgb_data[pos]
            results["lidar"] = input_data["rsu_{0}_lidar".format(self.id)][1]
            # print("RSU Frame: {}".format(input_data["rsu_{0}_lidar".format(self.id)][0]))
            if not is_train:
                bb_3d = self._get_3d_bbs(max_distance=self._3d_bb_distance)
                results["3d_bbs"] = bb_3d
                results["actors_data"] = self.collect_actor_data()
            # Other data
            cur_lidar_pose = carla.Location(x=self.lidar_pose.location.x+self._rsu_loc.x,
                                            y=self.lidar_pose.location.y+self._rsu_loc.y,
                                            z=self.lidar_pose.location.z+self._rsu_loc.z)
            # print([self._rsu_loc.x, self._rsu_loc.y, self._rsu_loc.z])
            self.intrinsics = get_camera_intrinsic(self._rgb_sensor_data)
            self.lidar2front = get_camera_extrinsic(self.camera_front_pose, self.lidar_pose)
            self.lidar2left = get_camera_extrinsic(self.camera_left_pose, self.lidar_pose)
            self.lidar2right = get_camera_extrinsic(self.camera_right_pose, self.lidar_pose)
            self.lidar2rear = get_camera_extrinsic(self.camera_rear_pose, self.lidar_pose)
            data = {
                "gps_x": -self._rsu_loc.y,
                "gps_y": self._rsu_loc.x,
                "x": self._rsu_loc.x,
                "y": self._rsu_loc.y,
                "theta": np.pi/2,
                "lidar_pose_x": cur_lidar_pose.x,
                "lidar_pose_y": cur_lidar_pose.y,
                "lidar_pose_z": cur_lidar_pose.z,
                "lidar_pose_gps_x": -cur_lidar_pose.y,
                "lidar_pose_gps_y": cur_lidar_pose.x,
                "camera_front_intrinsics": self.intrinsics,
                "camera_front_extrinsics": self.lidar2front,
                "camera_left_intrinsics": self.intrinsics,
                "camera_left_extrinsics": self.lidar2left,
                "camera_right_intrinsics": self.intrinsics,
                "camera_right_extrinsics": self.lidar2right,
                "camera_rear_intrinsics": self.intrinsics,
                "camera_rear_extrinsics": self.lidar2rear
            }
            results["measurements"] = data
            return results
        except:
            print('RSU process data error')
            return None

    def run_step(self, frame):
        """
        tick the data each step and store them in the correct directory.
        """
        if frame is not None and self.save_path is not None:
            # read the data from sensor
            input_data = self.sensor_interface.get_data()
            # preprocess the data, and store the data
            for cam in ["rgb", "depth"]:
                cam_data = {}
                for pos in ["front", "left", "right", "rear"]:
                    cam_data[pos] = cv2.cvtColor(
                        input_data["rsu_{0}_{1}_{2}".format(self.id,cam,pos)][1][:, :, :3],
                        cv2.COLOR_BGR2RGB
                    )
                    name_save = cam + "_" + pos
                    Image.fromarray(cam_data[pos]).save(
                        self.save_path_tmp / name_save / ("%04d.jpg" % frame)
                    )
            lidar_data = input_data["rsu_{0}_lidar".format(self.id)][1]
            # print("RSU Frame: {}".format(input_data["rsu_{0}_lidar".format(self.id)][0]))
            np.save(
                self.save_path_tmp / "lidar" / ("%04d.npy" % frame),
                lidar_data, allow_pickle=True,
            )
            lidar_semantic_front_data = input_data["rsu_{0}_lidar_semantic_front".format(self.id)][1]
            # print("RSU Frame: {}".format(input_data["rsu_{0}_lidar".format(self.id)][0]))
            np.save(
                self.save_path_tmp / "lidar_semantic_front" / ("%04d.npy" % frame),
                lidar_semantic_front_data, allow_pickle=True,
            )
            bb_3d = self._get_3d_bbs(max_distance=self._3d_bb_distance)
            np.save(
                self.save_path_tmp / "3d_bbs" / ("%04d.npy" % frame),
                bb_3d,
                allow_pickle=True,
            )
            # Other data
            cur_lidar_pose = carla.Location(x=self.lidar_pose.location.x+self._rsu_loc.x,
                                            y=self.lidar_pose.location.y+self._rsu_loc.y,
                                            z=self.lidar_pose.location.z+self._rsu_loc.z)
            # print([self._rsu_loc.x, self._rsu_loc.y, self._rsu_loc.z])
            self.intrinsics = get_camera_intrinsic(self._rgb_sensor_data)
            self.lidar2front = get_camera_extrinsic(self.camera_front_pose, self.lidar_pose)
            self.lidar2left = get_camera_extrinsic(self.camera_left_pose, self.lidar_pose)
            self.lidar2right = get_camera_extrinsic(self.camera_right_pose, self.lidar_pose)
            self.lidar2rear = get_camera_extrinsic(self.camera_rear_pose, self.lidar_pose)
            data = {
                "gps_x": -self._rsu_loc.y,
                "gps_y": self._rsu_loc.x,
                "x": self._rsu_loc.x,
                "y": self._rsu_loc.y,
                "theta": np.pi/2,
                "lidar_pose_x": cur_lidar_pose.x,
                "lidar_pose_y": cur_lidar_pose.y,
                "lidar_pose_z": cur_lidar_pose.z,
                "lidar_pose_gps_x": -cur_lidar_pose.y,
                "lidar_pose_gps_y": cur_lidar_pose.x,
                "camera_front_intrinsics": self.intrinsics,
                "camera_front_extrinsics": self.lidar2front,
                "camera_left_intrinsics": self.intrinsics,
                "camera_left_extrinsics": self.lidar2left,
                "camera_right_intrinsics": self.intrinsics,
                "camera_right_extrinsics": self.lidar2right,
                "camera_rear_intrinsics": self.intrinsics,
                "camera_rear_extrinsics": self.lidar2rear
            }
            measurements_file = self.save_path_tmp / "measurements" / ("%04d.json" % frame)
            f = open(measurements_file, "w")
            json.dump(data, f, indent=4)
            f.close()
            self.actors_data = self.collect_actor_data()
            lidar_pose = {"lidar_pose_x": data["lidar_pose_x"],
                          "lidar_pose_y": data["lidar_pose_y"],
                          "lidar_pose_z": data["lidar_pose_z"],
                          "theta": data["theta"]}
            self.actors_data, _ = process_lidar_visibility(self.actors_data, lidar_data, lidar_pose, change_actor_file=True)

            self.cur_camera_front_pose = carla.Location(x=self.camera_front_pose.location.x,
                                                y=self.camera_front_pose.location.y,
                                                z=self.camera_front_pose.location.z)        
            self.cur_camera_front_pose = carla.Location(x=self.cur_camera_front_pose.x+self._rsu_loc.x,
                                            y=self.cur_camera_front_pose.y+self._rsu_loc.y,
                                            z=self.cur_camera_front_pose.z+self._rsu_loc.z)

            bev_map = self.sg_lidar_2_bevmap(input_data)
            save_visibility_name = os.path.join(self.save_path_tmp,
                                                'bev_visibility',
                                                "%04d.png" % frame)
            cv2.imwrite(save_visibility_name, bev_map)


            self.env_actors_data = self.collect_env_actor_data()
            actors_data_file = self.save_path_tmp / "actors_data" / ("%04d.json" % frame)
            f = open(actors_data_file, "w")
            json.dump(self.actors_data, f, indent=4)
            f.close()
            env_actors_data_file = self.save_path_tmp / "env_actors_data" / ("%04d.json" % frame)
            f = open(env_actors_data_file, "w")
            json.dump(self.env_actors_data, f, indent=4)
            f.close()

        else:
            return
        
    def sg_lidar_2_bevmap(self, input_data):
        config = {'thresh':5,
                  'radius_meter':50,
                  'raster_size':[256, 256]
                  }

        # data = np.frombuffer(semantic_lidar, dtype=np.dtype([
        #     ('x', np.float32), ('y', np.float32), ('z', np.float32),
        #     ('CosAngle', np.float32), ('ObjIdx', np.uint32),
        #     ('ObjTag', np.uint32)]))

        # # (x, y, z, intensity)
        # points = np.array([data['x'], data['y'], data['z']]).T
        # obj_tag = np.array(data['ObjTag'])
        # obj_idx = np.array(data['ObjIdx'])
        # thresh = config['thresh']
        # # self.data = data
        # # self.frame = event.frame
        # # self.timestamp = event.timestamp

        # while obj_idx is None or obj_tag is None or \
        #         obj_idx.shape[0] != obj_tag.shape[0]:
        #     continue

        # # label 10 is the vehicle
        # vehicle_idx = obj_idx[obj_tag == 10]
        # # each individual instance id
        # vehicle_unique_id = list(np.unique(vehicle_idx))
        # vehicle_id_filter = []

        # for veh_id in vehicle_unique_id:
        #     if vehicle_idx[vehicle_idx == veh_id].shape[0] > thresh:
        #         vehicle_id_filter.append(veh_id)

        semantic_lidar_pose = {"lidar_pose_x": self.cur_camera_front_pose.x,
                        "lidar_pose_y": self.cur_camera_front_pose.y,
                        "lidar_pose_z": self.cur_camera_front_pose.z,
                        "theta": (self.camera_front_pose.rotation.yaw + 90)/180 * np.pi + np.pi/2}
        self.actors_data, vehicle_id_filter = process_lidar_visibility(self.actors_data, \
            input_data["rsu_{0}_lidar_semantic_front".format(self.id)][1], semantic_lidar_pose, change_actor_file=True, mode='camera', thresh = 5)

        self.radius_meter = config['radius_meter']
        self.raster_size = np.array(config['raster_size'])

        self.pixels_per_meter = self.raster_size[0] / (self.radius_meter * 2)

        meter_per_pixel = 1 / self.pixels_per_meter
        raster_radius = \
            float(np.linalg.norm(self.raster_size *
                                 np.array([meter_per_pixel,
                                           meter_per_pixel]))) / 2
        dynamic_agents = self.load_agents_world(raster_radius)
        final_agents = dynamic_agents

        # final_agents = agents_in_range(raster_radius,
        #                                     dynamic_agents)
        
        corner_list = []
        for agent_id, agent in final_agents.items():
            # in case we don't want to draw the cav itself
            # if not agent_id in vehicle_id_filter:
            #     continue
            agent_corner = self.generate_agent_area(agent['corners'])
            corner_list.append(agent_corner)

        self.vis_mask = 255 * np.zeros(
            shape=(self.raster_size[1], self.raster_size[0], 3),
            dtype=np.uint8)

        self.vis_mask = draw_agent(corner_list, self.vis_mask)

        return self.vis_mask
    
    def load_agents_world(self, max_distance=50):
        """
        Load all the dynamic agents info from server directly
        into a  dictionary.

        Returns
        -------
        The dictionary contains all agents info in the carla world.
        """

        vehicle_list = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
        walker_list = CarlaDataProvider.get_world().get_actors().filter("*walker.*")
        
        dynamic_agent_info = {}

        def read_actors_corner(agent_list, agent_info):
            for agent in agent_list:
                loc = agent.get_location()
                if loc.distance(self._rsu_loc) > 50:
                    continue
                agent_id = agent.id

                agent_transform = agent.get_transform()

                type_id = agent.type_id
                if not 'walker' in type_id:
                    agent_loc = [agent_transform.location.x,
                                agent_transform.location.y,
                                agent_transform.location.z - agent.bounding_box.extent.z, ]
                else:
                    agent_loc = [agent_transform.location.x,
                                agent_transform.location.y,
                                agent_transform.location.z, ]                

                agent_yaw = agent_transform.rotation.yaw

                # calculate 4 corners
                bb = agent.bounding_box.extent
                corners = [
                    carla.Location(x=-bb.x, y=-bb.y),
                    carla.Location(x=-bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=bb.y),
                    carla.Location(x=bb.x, y=-bb.y)
                ]
                # corners are originally in ego coordinate frame, convert to
                # world coordinate
                agent_transform.transform(corners)
                corners_reformat = [[x.x, x.y, x.z] for x in corners]

                agent_info[agent_id] = {'location': agent_loc,
                                                'yaw': agent_yaw,
                                                'corners': corners_reformat}
            return agent_info

        dynamic_agent_info = read_actors_corner(vehicle_list, dynamic_agent_info)
        dynamic_agent_info = read_actors_corner(walker_list, dynamic_agent_info)            

        return dynamic_agent_info
    
    def generate_agent_area(self, corners):
        """
        Convert the agent's bbx corners from world coordinates to
        rasterization coordinates.

        Parameters
        ----------
        corners : list
            The four corners of the agent's bbx under world coordinate.

        Returns
        -------
        agent four corners in image.
        """
        # (4, 3) numpy array
        corners = np.array(corners)
        # for homogeneous transformation
        corners = corners.T
        corners = np.r_[
            corners, [np.ones(corners.shape[1])]]
        # convert to ego's coordinate frame
        corners = world_to_sensor(corners, carla.Transform()).T
        corners = corners[:, :2]

        # switch x and y
        corners = corners[..., ::-1]
        # y revert
        corners[:, 1] = -corners[:, 1]

        corners[:, 0] = corners[:, 0] * self.pixels_per_meter + \
                        self.raster_size[0] // 2
        corners[:, 1] = corners[:, 1] * self.pixels_per_meter + \
                        self.raster_size[1] // 2

        # to make more precise polygon
        corner_area = cv2_subpixel(corners[:, :2])

        return corner_area