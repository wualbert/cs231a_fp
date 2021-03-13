from enum import Enum
from pyntcloud import PyntCloud
import param
import numpy as np
import os

def get_point_cloud_from_model(model_number):
    file = os.path.dirname(
        os.path.realpath(__file__)) + \
               '/../data/ycbv_models/models/obj_'+\
               str(model_number).zfill(6)+'.ply'
    cloud = PyntCloud.from_file(file)
    return cloud.points.values[:, :3]/1e3

def get_point_cloud_from_xyz(file):
    points = []
    with open(file, 'r') as f:
        for line in f:
            xyz = line.split()
            points.append(xyz)
    points = np.asarray(points, dtype=float)
    return points

def get_point_clouds_from_6881_model():
    drill_file = os.path.dirname(
        os.path.realpath(__file__)) + \
               '/../data/6881_models/'+\
               'drill_model'+'.npy'
    spatula_file = os.path.dirname(
        os.path.realpath(__file__)) + \
               '/../data/6881_models/'+\
               'spatula_model'+'.npy'
    drill = np.load(drill_file)
    spatula = np.load(spatula_file)
    return drill, spatula

# Project Specific Convenience Functions
def get_selected_clouds_for_project():
    object_names = param.selected_linemod_objects.keys()
    clouds = dict()
    for obj in object_names:
        obj = str(obj)
        file = param.linemod_path+'models/'+obj+'/'+\
               param.selected_linemod_objects[obj]+'.xyz'
        cloud = get_point_cloud_from_xyz(file)
        clouds[obj] = cloud
    return clouds

if __name__ == '__main__':
    get_point_cloud_from_model(1)