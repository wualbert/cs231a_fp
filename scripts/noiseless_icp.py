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

random_state = np.random.RandomState(0)
num_transforms = 10
dists = np.zeros([len(param.selected_models), num_transforms])

for model_index, model in enumerate(param.selected_models):
    point_cloud = io.get_point_cloud_from_model(model)
    for h_i in range(num_transforms):
        H = math_utils.generate_random_H(random_state=random_state)
        transformed_cloud = transform.transform_cloud(point_cloud, H)
        H_hat, d, i = icp.repeat_icp_until_convergence(point_cloud, transformed_cloud)
        davg = np.average(d)
        # print(f'H_hat:{H_hat}, distance:{davg}, i:{i}')
        dists[model_index, h_i] = davg
    print('done model ', model_index)
print('avg', np.average(dists, axis=1), 'std', np.std(dists, axis=1))
