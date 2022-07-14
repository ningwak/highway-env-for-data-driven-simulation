from collections import defaultdict
import itertools
import json
import random
import pdb
import numpy as np

from .data import SceneRow, TrackRow

import pdb

class Reader(object):
    """Read trajnet files.

    :param scene_type: None -> numpy.array, 'rows' -> TrackRow and SceneRow, 'paths': grouped rows (primary pedestrian first)
    """
    def __init__(self, input_file=None, scene_type=None):
        if scene_type is None:
            scene_type = 'paths'
        if input_file is None:
            input_file = 'DR_CHN_Merging_ZS.ndjson'
        self.scene_type = scene_type

        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()

        self.read_file(input_file)
        self.input_file = input_file
        
    def read_file(self, input_file):
        #pdb.set_trace()
        # self.scene_id = 38928
        # self.scene_id = 40015
        self.scene_id = 38100
        with open(input_file, 'r') as f:
            # print(f)
            for line in f:
                # print(line)
                line = json.loads(line)
                scene = line.get('scene')
                if scene is not None:
                    if scene['id'] == self.scene_id:
                        start =  scene['s']
                        end = scene['e']
                        row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'])
                        self.scenes_by_id[row.scene] = row

                track = line.get('track')
                if track is not None:
                    if track['f'] <= end and track['f'] >= start:
                        row = TrackRow(track['f'], track['p'], track['x'], track['y'])
                        self.tracks_by_frame[row.frame].append(row)
                    #pdb.set_trace()
                    continue

    @staticmethod
    def track_rows_to_paths(primary_pedestrian, track_rows):
        paths = defaultdict(list)
        for row in track_rows:
            paths[row.pedestrian].append(row)
        #if(track_rows[1].frame == 885):
         #   pdb.set_trace()
        # list of paths with the first path being the path of the primary pedestrian
        primary_path = paths[primary_pedestrian]
        other_paths = [path for ped_id, path in paths.items() if ped_id != primary_pedestrian]
        return [primary_path] + other_paths

    @staticmethod
    def paths_to_xy(paths):
        frames = [r.frame for r in
                  paths[0]]  # paths[0] is path of ped of interest. frames will be the list of frames in the scene
        pedestrians = [path[0].pedestrian for path in paths]  # pedestrians will be the list of pedestrians in the scene
        # print(pedestrians)

        frame_to_index = {frame: i for i, frame in enumerate(frames)}
        xy = np.full((len(pedestrians), len(frames), 2), np.nan)

        for ped_index, path in enumerate(paths):
            for indx, row in enumerate(path):
                # if(indx==16 and path[indx].pedestrian==20327):
                #   pdb.set_trace()
                if row.frame not in frame_to_index:
                    continue
                entry = xy[ped_index][frame_to_index[row.frame]]
                entry[1] = -row.x
                entry[0] = row.y
        # pdb.set_trace()
        return xy

    def scene(self, scene_id=None):
        if scene_id is None:
            scene_id = self.scene_id
        scene = self.scenes_by_id.get(scene_id)
        # remove address of the file and preserve the file name
        for i in range(len(self.input_file)):
            if (self.input_file[-i - 1] == '/'):
                break
        file_name = self.input_file[-i:-7]

        if scene is None:
            raise Exception('scene with that id not found')

        frames = range(scene.start, scene.end + 1)
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]
        # return as rows
        if self.scene_type == 'rows':
            return scene_id, scene.pedestrian, track_rows

        # return as paths
        paths = self.track_rows_to_paths(scene.pedestrian,
                                         track_rows)  # returns the paths(frames from first to end of a specific pedestrian) of different pedestrians, the first one is the path of ped on interest and then other ones.(so it is [[trajnettools.data.Rows of ped interest],[trajnettools.data.Row of next ped], ...]]
        if (paths[0][1].frame - paths[0][0].frame == 0):
            pdb.set_trace()
        if self.scene_type == 'paths':
            return file_name, scene_id, paths

        # return a numpy array
        return file_name, scene_id, self.paths_to_xy(paths)


