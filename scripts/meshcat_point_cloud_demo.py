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

random_state = np.random.RandomState(0)

for model_i in param.selected_models:
    point_cloud = io.get_point_cloud_from_model(model_i)
    vis = meshcat.Visualizer()
    vis.open()
    transformed_cloud, H = transform.generate_and_apply_random_transform(
        point_cloud, random_state)
    # H_hat, d, i = icp.icp(point_cloud, transformed_cloud)
    noisy_cloud = transform.apply_gaussian_noise(transformed_cloud, sigma=1e-2*np.ones(3))
    H_hat, d, i = icp.icp(point_cloud, noisy_cloud)

    # H_hat2, d2, i = icp.repeat_icp_until_convergence(point_cloud, transformed_cloud)
    print('i', i, np.average(d))
    visualize.visualize_point_cloud_meshcat(vis,
                                            [noisy_cloud,
                                     transform.transform_cloud(point_cloud,
                                                               H_hat),
                                     transformed_cloud]
                                            # transform.transform_cloud(point_cloud,
                                            #                           H_hat2)]
                                            )
