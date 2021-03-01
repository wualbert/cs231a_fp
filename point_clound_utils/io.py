from enum import Enum
from pyntcloud import PyntCloud
import numpy as np
import os

def get_point_cloud_from_model(model_number):
    file = os.path.dirname(
        os.path.realpath(__file__)) + \
               '/../data/ycbv_models/models/obj_'+\
               str(model_number).zfill(6)+'.ply'
    cloud = PyntCloud.from_file(file)
    return cloud.points.values[:, :3]/1e3

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

if __name__ == '__main__':
    get_point_cloud_from_model(1)