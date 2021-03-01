import icp.icp as icp
import param
import point_clound_utils.io as io
import point_clound_utils.transform as transform
import point_clound_utils.visualize as visualize
import numpy as np
import os
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

for model_i in param.selected_models:
    point_cloud = io.get_point_cloud_from_model(model_i)
    vis = meshcat.Visualizer()
    vis.open()
    # H = np.eye(4)
    # H[:3, -1] = np.array([1., 0., 0.])
    # transformed_cloud = transform.transform_cloud(point_cloud, H)
    # T, distances, i = icp.icp(point_cloud, transformed_cloud)
    # print(f'T:{T}, distance:{np.sum(distances)}, i:{i}')
    #
    # gaussian_cloud = transform.apply_gaussian_noise(point_cloud)
    # T, distances, i = icp.icp(point_cloud, gaussian_cloud)
    # print(f'T:{T}, distance:{np.sum(distances)}, i:{i}')
    #
    # icp_cloud = transform.transform_cloud(point_cloud, T)
    visualize.visualize_point_cloud(vis, [point_cloud])
