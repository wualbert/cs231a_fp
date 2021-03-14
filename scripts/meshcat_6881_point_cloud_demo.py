import point_clound_utils.io as io
import point_clound_utils.visualize as visualize
import numpy as np
import os
import time
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

drill, spatula = io.get_point_clouds_from_6881_model()
vis = meshcat.Visualizer()
vis.open()
print(drill.shape)
visualize.visualize_point_cloud_meshcat(vis, [drill])
