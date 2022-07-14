import numpy as np
import math

from highway_env.envs.reader import Reader

from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class DatasetMergeEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    COLLISION_REWARD: float = -100
    RIGHT_LANE_REWARD: float = 0.1
    HIGH_SPEED_REWARD: float = 0.2
    MERGING_SPEED_REWARD: float = -0.5
    LANE_CHANGE_REWARD: float = -0.05
    CLOSE_DISTANCE_REWARD: float = -2

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.LANE_CHANGE_REWARD,
                         1: 0,
                         2: self.LANE_CHANGE_REWARD,
                         3: 0,
                         4: 0}
        # print("vehicle crashed:", self.vehicle.crashed)
        front_vehicle, rear_vehicle = self.vehicle.road.neighbour_vehicles(self.vehicle, self.vehicle.target_lane_index)

        reward = self.COLLISION_REWARD * self.vehicle.crashed \
                 + self.RIGHT_LANE_REWARD * self.vehicle.lane_index[2] / 1 \
                 + self.HIGH_SPEED_REWARD * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)

        if front_vehicle:
            d = self.vehicle.lane_distance_to(front_vehicle)
            reward = reward + self.CLOSE_DISTANCE_REWARD / d
        # print("reward:", reward)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.MERGING_SPEED_REWARD * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        return utils.lmap(action_reward[action] + reward,
                          [self.COLLISION_REWARD + self.MERGING_SPEED_REWARD,
                            self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])
        '''if isinstance(action, int):
            return utils.lmap(action_reward[action] + reward,
                              [self.COLLISION_REWARD + self.MERGING_SPEED_REWARD,
                               self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                              [0, 1])
        else:
            return utils.lmap(self.LANE_CHANGE_REWARD * abs(action[1]) + reward,
                              [self.COLLISION_REWARD + self.MERGING_SPEED_REWARD,
                               self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
                          [0, 1])'''

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or self.vehicle.position[0] > 370

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        offset = [982.9, 984.7]
        scale = 8.5
        net.add_lane("a", "b", StraightLane([1489 / scale + offset[1], 158 / scale + offset[0]], [1157 / scale + offset[1], 227 / scale + offset[0]], line_types=[s, c]))
        net.add_lane("a", "b", StraightLane([1489 / scale + offset[1], 188 / scale + offset[0]], [1157 / scale + offset[1], 257 / scale + offset[0]], line_types=[c, n]))

        net.add_lane("b", "c", StraightLane([1157 / scale + offset[1], 227 / scale + offset[0]],
                                            [1025 / scale + offset[1], 243 / scale + offset[0]], 35 / scale, line_types=[s, c]))
        net.add_lane("b", "c", StraightLane([1157 / scale + offset[1], 257 / scale + offset[0]],
                                            [1025 / scale + offset[1], 273 / scale + offset[0]], 35 / scale, line_types=[c, n]))

        net.add_lane("c", "d", StraightLane([1025 / scale + offset[1], 243 / scale + offset[0]],
                                            [852 / scale + offset[1], 254 / scale + offset[0]], 35 / scale, line_types=[s, c]))
        net.add_lane("c", "d", StraightLane([1025 / scale + offset[1], 273 / scale + offset[0]],
                                            [851 / scale + offset[1], 283 / scale + offset[0]], 35 / scale, line_types=[c, n]))

        net.add_lane("d", "e", StraightLane([852 / scale + offset[1], 254 / scale + offset[0]],
                                            [647 / scale + offset[1], 248 / scale + offset[0]], 35 / scale, line_types=[s, c]))
        net.add_lane("d", "e", StraightLane([851 / scale + offset[1], 283 / scale + offset[0]],
                                            [646 / scale + offset[1], 277 / scale + offset[0]], 35 / scale, line_types=[c, n]))

        net.add_lane("e", "f", StraightLane([646 / scale + offset[1], 211 / scale + offset[0]],
                                            [505 / scale + offset[1], 197 / scale + offset[0]], 35 / scale, line_types=[n, c]))
        net.add_lane("e", "f", StraightLane([647 / scale + offset[1], 248 / scale + offset[0]],
                                            [506 / scale + offset[1], 234 / scale + offset[0]], 35 / scale, line_types=[s, s]))
        net.add_lane("e", "f", StraightLane([646 / scale + offset[1], 277 / scale + offset[0]],
                                            [488 / scale + offset[1], 266 / scale + offset[0]], 35 / scale, line_types=[c, n]))

        net.add_lane("f", "g", StraightLane([505 / scale + offset[1], 197 / scale + offset[0]],
                                            [137 / scale + offset[1], 125 / scale + offset[0]], 35 / scale, line_types=[n, c]))
        net.add_lane("f", "g", StraightLane([506 / scale + offset[1], 234 / scale + offset[0]],
                                            [132 / scale + offset[1], 154 / scale + offset[0]], 35 / scale, line_types=[s, s]))
        net.add_lane("f", "g", StraightLane([488 / scale + offset[1], 266 / scale + offset[0]],
                                            [125 / scale + offset[1], 185 / scale + offset[0]], 35 / scale, line_types=[c, n]))

        net.add_lane("g", "h", StraightLane([137 / scale + offset[1], 125 / scale + offset[0]],
                                            [1 / scale + offset[1], 65 / scale + offset[0]], 35 / scale, line_types=[n, c]))
        net.add_lane("g", "h", StraightLane([132 / scale + offset[1], 154 / scale + offset[0]],
                                            [1 / scale + offset[1], 105 / scale + offset[0]], 35 / scale, line_types=[s, s]))
        net.add_lane("g", "h", StraightLane([125 / scale + offset[1], 185 / scale + offset[0]],
                                            [1 / scale + offset[1], 140 / scale + offset[0]], 35 / scale, line_types=[c, n]))
        # Merging lane
        net.add_lane("i", "j", StraightLane([1489 / scale + offset[1], 61 / scale + offset[0]],
                                            [927 / scale + offset[1], 184 / scale + offset[0]], 31 / scale, line_types=[n, c]))
        net.add_lane("i", "j", StraightLane([1489 / scale + offset[1], 92 / scale + offset[0]],
                                            [923 / scale + offset[1], 204 / scale + offset[0]], 31 / scale, line_types=[c, s]))

        net.add_lane("j", "k", StraightLane([927 / scale + offset[1], 184 / scale + offset[0]],
                                            [833 / scale + offset[1], 193 / scale + offset[0]], 31 / scale, line_types=[n, c]))
        net.add_lane("j", "k", StraightLane([923 / scale + offset[1], 204 / scale + offset[0]],
                                            [836 / scale + offset[1], 212 / scale + offset[0]], 31 / scale, line_types=[c, s]))

        net.add_lane("k", "l", StraightLane([833 / scale + offset[1], 193 / scale + offset[0]],
                                            [727 / scale + offset[1], 209 / scale + offset[0]], line_types=[n, c]))
        net.add_lane("k", "l", StraightLane([836 / scale + offset[1], 212 / scale + offset[0]],
                                            [727 / scale + offset[1], 209 / scale + offset[0]], line_types=[c, s]))

        net.add_lane("l", "e", StraightLane([727 / scale + offset[1], 209 / scale + offset[0]],
                                            [646 / scale + offset[1], 211 / scale + offset[0]], 35 / scale, line_types=[c, c]))
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        '''ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(30, 0),
                                                     speed=3)'''
        # ego_vehicle = self.action_type.vehicle_class(road,
        #                                              road.network.get_lane(("b", "c", 1)).position(1, 0), heading = 2,
        #                                              speed=5)
        # ego_vehicle.target_speed = 3
        # road.vehicles.append(ego_vehicle)
        # print(ego_vehicle.position)

        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_vehicles_type = utils.class_from_path("highway_env.vehicle.kinematics.DirectSetVehicle")
        '''position_list = [[1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],
                         [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010], [1100, 1010],



                         ]'''
        reader = Reader(scene_type='paths')
        _, _, paths = reader.scene()
        ped_id = paths[0][0].pedestrian
        xy = Reader.paths_to_xy(paths)
        # print(paths)
        # print(xy)
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     xy[0][0], heading = 2,
                                                     speed=5)
        ego_vehicle.target_speed = 3
        road.vehicles.append(ego_vehicle)
        for i in range(1, len(xy)):
            position_list = xy[i]
            if not math.isnan(position_list[0][0]):
                starting_point = position_list[0]
            else:
                starting_point = [10000, 10000]
            road.vehicles.append(
                other_vehicles_type(road, starting_point, position_list))

        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), position_list, speed=5))

        '''merging_v = other_vehicles_type(road, road.network.get_lane(("i", "j", 0)).position(20, 0), speed=5)
        merging_v.target_speed = 3
        road.vehicles.append(merging_v)'''

        self.vehicle = ego_vehicle


register(
    id='dataset-merge-v0',
    entry_point='highway_env.envs:DatasetMergeEnv',
)
