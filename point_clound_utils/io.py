from enum import Enum
import open3d as o3d
from pyntcloud import PyntCloud
import param
import numpy as np
import os
import bop_toolkit_lib.inout as bop_io
import linecache
import cv2

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
def load_clouds_from_selected_models(open3d=True,
                                     object_names=None):
    if object_names is None:
        object_names = param.selected_linemod_objects.keys()
    clouds = dict()
    for obj in object_names:
        obj_cap = str(obj).capitalize()
        img_number = str(param.selected_linemod_objects[obj_cap]).zfill(3)
        file = param.linemod_path+'models/'+obj+'/'+\
               img_number+'.xyz'
        if not open3d:
            cloud = get_point_cloud_from_xyz(file)
        else:
            cloud = o3d.io.read_point_cloud(file)
        clouds[obj.lower()] = cloud
    return clouds


def load_rgb_and_d(image_number=0, open3d=False):
    img_file = param.linemod_path+'RGB-D/rgb_noseg/color_'+\
               str(image_number).zfill(5)+'.png'
    depth_file = param.linemod_path+'RGB-D/depth_noseg/depth_'+\
               str(image_number).zfill(5)+'.png'
    img = o3d.io.read_image(img_file)
    depth = o3d.io.read_image(depth_file)
    if open3d:
        return img, depth
    else:
        return np.array(img), np.array(depth)/1000.

def load_cloud_from_selected_image(image_number=0, intrinsics=None):
    if intrinsics is None:
        intrinsics = load_camera_intrinsics_open3d()
    img,depth=load_rgb_and_d(image_number,open3d=True)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        img, depth,
        depth_scale=1000.0)#see camera.json
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd

def load_camera_intrinsics_open3d():
    path = param.linemod_path+'/Lmo/camera.json'
    bop_intrinsics = bop_io.load_cam_params(path)
    open3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    open3d_intrinsics.set_intrinsics(width=bop_intrinsics['im_size'][0],
                                     height=bop_intrinsics['im_size'][1],
                                     fx=bop_intrinsics['K'][0,0],
                                     fy=bop_intrinsics['K'][1,1],
                                     cx=bop_intrinsics['K'][0,2],
                                     cy=bop_intrinsics['K'][1,2],
                                     )
    return open3d_intrinsics

def load_segpose_prediction(obj_name, image_number, out_i = ''):
    obj_name = obj_name.lower()
    image_number = str(image_number).zfill(4)
    file = param.segpose_out_path+out_i+'/'+obj_name+'/'+image_number+'.txt'
    RT = np.loadtxt(file)
    return RT

def load_ground_truth_pose(obj_name, image_number, out_i = ''):
    obj_name = obj_name.capitalize()
    image_number = str(image_number).zfill(5)
    file = param.linemod_path+'poses/'+obj_name+'/info_'+image_number+'.txt'
    R = []
    for i in range(5,8):
        line = linecache.getline(file, i).split()
        R.append(line)
    R = np.array(R, dtype=float)
    T = np.array([linecache.getline(file, 9).split()],dtype=float)
    flip = -np.eye(3)
    flip[0,0] *= -1
    return flip @ np.hstack([R,T.T])

if __name__ == '__main__':
    get_point_cloud_from_model(1)