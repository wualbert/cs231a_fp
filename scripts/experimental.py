import param
import math_utils
import icp.icp as icp
import point_clound_utils.io as io
import point_clound_utils.transform as transform
import point_clound_utils.visualize as visualize
import numpy as np
import os
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

import open3d

pcd = io.load_cloud_from_selected_image_id()
model_cloud = io.load_clouds_from_selected_models()['Driller']


threshold = 0.3
RT = io.load_segpose_prediction('Driller', 0)
RT_gt = io.load_ground_truth_pose('Driller', 0)
# trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
#                          [-0.139, 0.967, -0.215, 0.7],
#                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
trans = np.eye(4)
trans[:3,:] = RT
reg_p2p = open3d.registration.registration_icp(
    model_cloud, pcd, threshold, trans)

# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 1.5
#     print(":: RANSAC registration on downsampled point clouds.")
#     print("   Since the downsampling voxel size is %.3f," % voxel_size)
#     print("   we use a liberal distance threshold %.3f." % distance_threshold)
#     result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
#         open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         4, [
#             open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 0.9),
#             open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], open3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
#     return result

print(reg_p2p.transformation)

visualize.draw_registration_result_open3d(model_cloud, pcd, reg_p2p.transformation)
# open3d.draw_geometries([pcd])

# vis = meshcat.Visualizer()
# vis.open()
# clouds = io.get_clouds_from_selected_models()
# visualize.visualize_point_cloud(vis, clouds.values())
# time.sleep(100)

