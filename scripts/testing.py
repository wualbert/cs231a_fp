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

vis = meshcat.Visualizer()
vis.open()
clouds = io.get_selected_clouds_for_project()
visualize.visualize_point_cloud(vis, clouds.values())
time.sleep(100)