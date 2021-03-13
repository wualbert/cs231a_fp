import param
import math_utils
import visualization_utils as viz
import icp.icp as icp
import point_clound_utils.io as io
import point_clound_utils.transform as transform
import point_clound_utils.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

random_state = np.random.RandomState(0)
num_transforms = 10
sigmas = 0.001*(5*np.arange(5))
# colors = ['c', 'm', 'y', 'k', 'g']
colors = ['k']*5
labels = ['Coffee Can', 'Mustard Container', 'Banana', 'Power Drill', 'Scissors']
for model_index, model in enumerate(param.selected_models):
    fig, ax = plt.subplots()
    point_cloud = io.get_point_cloud_from_model(model)
    y = np.zeros(5)
    yerr = np.zeros(5)
    ymin = np.zeros(5)
    ymax = np.zeros(5)
    for sigma_index, sigma in enumerate(sigmas):
        dists = np.zeros(num_transforms)
        for h_i in range(num_transforms):
            H = math_utils.generate_random_H(random_state=random_state)
            transformed_cloud = transform.transform_cloud(point_cloud, H)
            noisy_cloud = transform.apply_gaussian_noise(transformed_cloud, sigma=sigma*np.ones(3))
            H_hat, d_drop, i = icp.icp(point_cloud, transformed_cloud)
            # Compute the distance between the transformed cloud and noisy cloud
            estimated_transformed_cloud = transform.transform_cloud(point_cloud, H_hat)
            point_dist, _ = icp.nearest_neighbor(transformed_cloud, estimated_transformed_cloud)
            add = np.average(point_dist)
            # print(f'H_hat:{H_hat}, distance:{add}, i:{i}')
            dists[h_i] = add
        dists *= 1e2
        y[sigma_index] = np.average(dists)
        yerr[sigma_index] = np.std(dists)
        ymin[sigma_index] = np.min(dists)
        ymax[sigma_index] = np.max(dists)
    print('done model ', model_index)
    viz.plot_error_with_min_max(sigmas*1e2, y, yerr, ymin, ymax,
                                fig, ax, colors[model_index],
                                label=labels[model_index])
    ax.legend()
    ax.set_xlabel('Gaussian noise $\sigma$ (cm)')
    ax.set_ylabel('ADD(cm)')
    plt.savefig(f'gaussian_{labels[model_index]}.png', dpi=300)
    plt.show()
