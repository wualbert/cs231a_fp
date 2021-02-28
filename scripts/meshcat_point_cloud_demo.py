import point_clound_utils.model as model
import point_clound_utils.visualize as visualize
import numpy as np
import os
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

point_cloud = model.get_point_cloud_from_model(1)
vis = meshcat.Visualizer()
vis.open()
visualize.visualize_point_cloud(vis, [point_cloud.points.values])
