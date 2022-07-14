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


class DatasetMergeM1Env(AbstractEnv):

    """
    A highway merge negotiation environment.

    The layout is based on DR_CHN_Merging_ZS in Interaction Dataset. It is directly based on the .osm file.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    This env is using generated surrounding vehicles
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

        return utils.lmap(reward,
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

        scale = 1
        offset = [0, 1]

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([(1146.10712 + 1146.40984) / 2, -(965.15632 + 961.33538) / 2],
                                            [(1118.04010 + 1118.52411) / 2, -(959.86428 + 956.04654) / 2], 4.68773, line_types=[s, c]))
        net.add_lane("a", "b", StraightLane([(1146.58597 + 1146.40984) / 2, -(957.39920 + 961.33538) / 2],
                                            [(1119.20271 + 1118.52411) / 2, -(952.22416 + 956.04654) / 2], 4.68773, line_types=[c, n]))

        net.add_lane("b", "c", StraightLane([(1118.04010 + 1118.52411) / 2, -(959.86428 + 956.04654) / 2],
                                            [(1107.86842 + 1107.88710) / 2, -(958.65705 + 954.71964) / 2], 4.68773, line_types=[s, c]))
        net.add_lane("b", "c", StraightLane([(1119.20271 + 1118.52411) / 2, -(952.22416 + 956.04654) / 2],
                                            [(1108.97047 + 1107.88710) / 2, -(950.54743 + 954.71964) / 2], 4.68773, line_types=[c, n]))

        net.add_lane("c", "d", StraightLane([(1107.86842 + 1107.88710) / 2, -(958.65705 + 954.71964) / 2],
                                            [(1089.28267 + 1089.19981) / 2, -(957.31980 + 953.11457) / 2], 4.68773, line_types=[s, c]))
        net.add_lane("c", "d", StraightLane([(1108.97047 + 1107.88710) / 2, -(950.54743 + 954.71964) / 2],
                                            [(1089.02694 + 1089.19981) / 2, -(948.87071 + 953.11457) / 2], 4.68773, line_types=[c, n]))

        net.add_lane("d", "e", StraightLane([(1089.28267 + 1089.19981) / 2, -(957.31980 + 953.11457) / 2],
                                            [(1063.57299 + 1063.91892) / 2, -(957.39410 + 952.72780) / 2], 4.68773, line_types=[s, c]))
        net.add_lane("d", "e", StraightLane([(1089.02694 + 1089.19981) / 2, -(948.87071 + 953.11457) / 2],
                                            [(1063.72760 + 1063.91892) / 2, -(948.95352 + 952.72780) / 2], 4.68773, line_types=[c, n]))

        net.add_lane("e", "f", StraightLane([(1063.57299 + 1063.87215) / 2, -(957.39410 + 960.96007) / 2],
                                            [(1043.49140 + 1039.60224) / 2, -(958.58275 + 964.04313) / 2], 3.56597, line_types=[s, c]))
        net.add_lane("e", "f", StraightLane([(1063.57299 + 1063.91892) / 2, -(957.39410 + 952.72780) / 2],
                                            [(1043.49140 + 1046.78816) / 2, -(958.58275 + 953.91575) / 2], 4.68773, line_types=[s, n]))
        net.add_lane("e", "f", StraightLane([(1063.72760 + 1063.91892) / 2, -(948.95352 + 952.72780) / 2],
                                            [(1046.59742 + 1046.78816) / 2, -(950.07132 + 953.91575) / 2], 4.68773, line_types=[c, n]))

        net.add_lane("f", "g", StraightLane([(1043.49140 + 1039.60224) / 2, -(958.58275 + 964.04313) / 2],
                                            [(1024.26416 + 1025.21288) / 2, -(961.88454 + 966.55043) / 2], 4.68773, line_types=[s, c]))
        net.add_lane("f", "g", StraightLane([(1043.49140 + 1046.78816) / 2, -(958.58275 + 953.91575) / 2],
                                            [(1024.26416 + 1023.36523) / 2, -(961.88454 + 957.61713) / 2], 4.68773, line_types=[s, n]))
        net.add_lane("f", "g", StraightLane([(1046.59742 + 1046.78816) / 2, -(950.07132 + 953.91575) / 2],
                                            [(1022.50679 + 1023.36523) / 2, -(953.71453 + 957.61713) / 2], 4.68773, line_types=[c, n]))

        net.add_lane("g", "h", StraightLane([(1024.26416 + 1025.21288) / 2, -(961.88454 + 966.55043) / 2],
                                            [(998.79047 + 999.24068) / 2, -(967.98732 + 972.02941) / 2], 4.68773, line_types=[s, c]))
        net.add_lane("g", "h", StraightLane([(1024.26416 + 1023.36523) / 2, -(961.88454 + 957.61713) / 2],
                                            [(998.79047 + 998.21777) / 2, -(967.98732 + 963.83533) / 2], 4.68773, line_types=[s, n]))
        net.add_lane("g", "h", StraightLane([(1022.50679 + 1023.36523) / 2, -(953.71453 + 957.61713) / 2],
                                            [(997.41585 + 998.21777) / 2, -(960.09017 + 963.83533) / 2], 4.68773, line_types=[c, n]))
        # Merging lane
        net.add_lane("i", "j", StraightLane([(1146.22459 + 1146.31490) / 2, -(974.52698 + 971.74549) / 2],
                                            [(1107.25139 + 1107.43837) / 2, -(966.86621 + 964.15458) / 2], 2.78149, line_types=[s, c]))
        net.add_lane("i", "j", StraightLane([(1146.10569 + 1146.31490) / 2, -(968.59348 + 971.74549) / 2],
                                            [(1107.75622 + 1107.43837) / 2, -(961.01581 + 964.15458) / 2], 2.78149, line_types=[c, n]))

        net.add_lane("j", "k", StraightLane([(1107.25139 + 1107.43837) / 2, -(966.86621 + 964.15458) / 2],
                                            [(1088.85260 + 1089.03954) / 2, -(964.74893 + 961.85161) / 2], 2.71163, line_types=[s, c]))
        net.add_lane("j", "k", StraightLane([(1107.75622 + 1107.43837) / 2, -(961.01581 + 964.15458) / 2],
                                            [(1094.16282 + 1089.03954) / 2, -(958.61991 + 961.85161) / 2], 3.13877, line_types=[c, n]))

        net.add_lane("k", "l", StraightLane([(1088.85260 + 1089.03954) / 2, -(964.74893 + 961.85161) / 2],
                                            [(1080.22909 + 1080.25129) / 2, -(963.74919 + 961.06428) / 2], 2.71163, line_types=[s, c]))
        net.add_lane("k", "l", StraightLane([(1094.16282 + 1089.03954) / 2, -(958.61991 + 961.85161) / 2],
                                            [(1080.12895 + 1080.25129) / 2, -(957.40215 + 961.06428) / 2], 3.66213, line_types=[c, n]))

        net.add_lane("l", "e", StraightLane([(1080.22909 + 1080.25129) / 2, -(963.74919 + 961.06428) / 2],
                                            [(1063.87215 + 1063.57299) / 2, -(960.96007 + 957.39410) / 2], 2.71163,
                                            line_types=[n, c]))
        net.add_lane("l", "e", StraightLane([(1080.12895 + 1080.25129) / 2, -(957.40215 + 961.06428) / 2],
                                            [(1063.87215 + 1063.57299) / 2, -(960.96007 + 957.39410) / 2], 3.66213,
                                            line_types=[c, s]))
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
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("b", "c", 0)).position(1, 0), heading = 2,
                                                     speed=5)
        ego_vehicle.target_speed = 5
        road.vehicles.append(ego_vehicle)
        # print(ego_vehicle.position)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=5))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("b", "c", 1)).position(5, 0), speed=5))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("d", "e", 1)).position(1, 0), speed=5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("i", "j", 0)).position(20, 0), speed=5)
        merging_v.target_speed = 3
        road.vehicles.append(merging_v)

        self.vehicle = ego_vehicle


register(
    id='dataset-merge-m-v1',
    entry_point='highway_env.envs:DatasetMergeM1Env',
)
