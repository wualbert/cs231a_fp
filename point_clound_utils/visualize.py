# Adapted from PSet 3 iterative_closes_point.py of MIT 6.881 Fall 2018
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np


def make_meshcat_color_array(N, r, g, b):
    '''
    Construct a color array to visualize a point cloud in meshcat

    Args:
        N: int. number of points to generate. Must be >= number of points in the
            point cloud to color
        r: float. The red value of the points, 0.0 <= r <= 1.0
        g: float. The green value of the points, 0.0 <= g <= 1.0
        b: float. The blue value of the points, 0.0 <= b <= 1.0

    Returns:
        3xN numpy array of the same color
    '''
    color = np.zeros((3, N))
    color[0, :] = r
    color[1, :] = g
    color[2, :] = b

    return color


rgb_sequences = [(0.5, 0., 0.), (0., 0., 0.5), (1., 1., 0.)]

def visualize_point_cloud(meshcat_vis, clouds:np.ndarray):
    '''
    Visualize the ground truth (red), observation (blue), and transformed
    (yellow) point clouds in meshcat.

    Args:
        meschat_vis: an instance of a meshcat visualizer
        scene: an Nx3 numpy array representing scene point cloud
        model: an Mx3 numpy array representing model point cloud
        X_GO: 4x4 numpy array of the homogeneous transformation from the
            scene point cloud to the model point cloud.
    '''

    meshcat_vis['model'].delete()
    meshcat_vis['observations'].delete()
    meshcat_vis['transformed_observations'].delete()

    for i, cloud in enumerate(clouds):
        # Make meshcat color arrays.
        N = cloud.shape[0]
        if i<3:
            color_arr = make_meshcat_color_array(N, *rgb_sequences[i])
        else:
            raise NotImplementedError

        # Create red and blue meshcat point clouds for visualization.
        cloud_meshcat = g.PointCloud(cloud[:,:3].T, color_arr, size=0.01)

        meshcat_vis[str(i)].set_object(cloud_meshcat)
