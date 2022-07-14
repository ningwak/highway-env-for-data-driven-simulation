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


class DatasetMergeMEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The layout is based on DR_CHN_Merging_ZS in Interaction Dataset. It is directly based on the .osm file.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.config["right_lane_reward"] * self.vehicle.lane_index[2] / 1 \
                 + self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.SPEED_COUNT - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                reward += self.config["merging_speed_reward"] * \
                          (vehicle.target_speed - vehicle.speed) / vehicle.target_speed

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

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

        # net.add_lane("l", "e", StraightLane([(1080.22909 + 1080.25129) / 2, -(963.74919 + 961.06428) / 2],
        #                                     [(1063.87215 + 1063.57299) / 2, -(960.96007 + 957.39410) / 2], 2.71163,
        #                                     line_types=[n, c]))
        lle = StraightLane([(1080.22909 + 1080.25129) / 2, -(963.74919 + 961.06428) / 2],
                           [(1063.87215 + 1063.57299) / 2, -(960.96007 + 963.69410) / 2], 2.71163,
                           line_types=[n, c])
        net.add_lane("l", "e", lle)
        net.add_lane("l", "e", StraightLane([(1080.12895 + 1080.25129) / 2, -(957.40215 + 961.06428) / 2],
                                            [(1063.87215 + 1063.57299) / 2, -(960.96007 + 957.39410) / 2], 3.66213,
                                            line_types=[c, s]))
        
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lle.position(16.5, 0)))
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
        real_ego_vehicles_type = utils.class_from_path("highway_env.vehicle.behavior.IDMVehicle")
        other_vehicles_type = utils.class_from_path("highway_env.vehicle.behavior.IDMVehicle")
        other_vehicles_type_direct = utils.class_from_path("highway_env.vehicle.kinematics.DirectSetVehicle")
        
        reader = Reader(scene_type='paths')
        _, _, paths = reader.scene()
        ped_id = paths[0][0].pedestrian
        xy = Reader.paths_to_xy(paths)
        # print(paths)
        # print(xy)
        # ego_vehicle = self.action_type.vehicle_class(road,
        #                                              [xy[0][4][0], xy[0][4][1]], heading = np.pi,
        #                                              speed=0)
        # ego_vehicle = self.action_type.vehicle_class(road,
        #                                              [xy[0][4][0] - 140, xy[0][4][1] + 10], heading=np.pi,
        #                                              speed=0)
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     [xy[0][4][0] - 70, xy[0][4][1] + 10], heading=np.pi,
                                                     speed=0)
        ego_vehicle.target_speed = 0
        road.vehicles.append(ego_vehicle)
        # print(xy[0][:])
        # print(np.pi + (xy[0][5][1] - xy[0][4][1]) / (xy[0][5][0] - xy[0][4][0]))
        # road.vehicles.append(
        #     real_ego_vehicles_type(road, xy[0][4], heading=np.pi + (xy[0][5][1] - xy[0][4][1]) / (xy[0][5][0] - xy[0][4][0]),
        #                            speed=2*np.sqrt((xy[0][4][0] - xy[0][3][0]) ** 2 + (xy[0][4][1] - xy[0][3][1]) ** 2)))

        # for i in range(1, len(xy)):
        #     print('xy', i, xy[i][4])
        o = 0
        # for otest in range(0, len(xy)): 
        #     print('xy', otest, xy[otest][4])
        self.reference = xy[o][4:]
        for i in range(0, len(xy)):
        # TODO: comment the next line and use the former one
        # for i in range(1, 2):
            position_list = xy[i][4:]
            obss = xy[i][0:4]
            if xy[i][4][0] - xy[i][0][0] >= 0:
                continue
            if np.isnan(xy[i][4][0] - xy[i][0][0]):
                continue
            if not math.isnan(xy[i][4][0]):
                starting_point = xy[i][4]
                # print(position_list[1])
            else:
                starting_point = [10000, 10000]
            # road.vehicles.append(
            #     other_vehicles_type(road, starting_point, position_list))
            if np.isnan(np.pi + (xy[i][5][1] - xy[i][4][1]) / (xy[i][5][0] - xy[i][4][0])):
                heading = np.pi
            else:
                heading = heading = np.pi + (xy[i][5][1] - xy[i][4][1]) / (xy[i][5][0] - xy[i][4][0])
            if np.isnan(np.sqrt((xy[i][4][0] - xy[i][0][0]) ** 2 + (xy[i][4][1] - xy[i][0][1]) ** 2)):
                speed = 3
                target_speed = 3
            else:
                speed = 2 * np.sqrt((xy[i][4][0] - xy[i][3][0]) ** 2 + (xy[i][4][1] - xy[i][3][1]) ** 2)
                target_speed = 1/2 * np.sqrt((xy[i][4][0] - xy[i][0][0]) ** 2 + (xy[i][4][1] - xy[i][0][1]) ** 2)
            # elif np.isnan(np.sqrt((xy[i][4][0] - xy[i][0][0]) ** 2 + (xy[i][4][1] - xy[i][0][1]) ** 2)):
            #     speed = 2 * np.sqrt((xy[i][4][0] - xy[i][3][0]) ** 2 + (xy[i][4][1] - xy[i][3][1]) ** 2)
            #     target_speed = speed
            # else:
            #     speed = 0.5 * np.sqrt((xy[i][4][0] - xy[i][0][0]) ** 2 + (xy[i][4][1] - xy[i][0][1]) ** 2)
            #     target_speed = speed
            # print(target_speed)
            # print(heading, speed)
            road.vehicles.append(
                other_vehicles_type(road, starting_point, heading=heading, speed=3, target_speed=target_speed, obss=obss))
            # if i == o:
            #     road.vehicles.append(
            #         other_vehicles_type(road, starting_point, heading=-heading, speed=10, target_speed=target_speed, obss=obss))
            #     print('o', starting_point)
            # else:
            #     road.vehicles.append(
            #         other_vehicles_type_direct(road, starting_point, position_list, speed=5, obss=obss))
            #     print(starting_point)

        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), position_list, speed=5))

        '''merging_v = other_vehicles_type(road, road.network.get_lane(("i", "j", 0)).position(20, 0), speed=5)
        merging_v.target_speed = 3
        road.vehicles.append(merging_v)'''

        self.vehicle = ego_vehicle


register(
    id='dataset-merge-m-v0',
    entry_point='highway_env.envs:DatasetMergeMEnv',
)
