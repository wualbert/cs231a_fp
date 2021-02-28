from enum import Enum
from pyntcloud import PyntCloud
import os

def get_point_cloud_from_model(model_number):
    file = os.path.dirname(
        os.path.realpath(__file__)) + \
               '/../data/ycbv_models/models/obj_'+\
               str(model_number).zfill(6)+'.ply'
    cloud = PyntCloud.from_file(file)
    return cloud

if __name__ == '__main__':
    get_point_cloud_from_model(1)