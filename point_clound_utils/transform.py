import numpy as np

def transform_cloud(cloud, H:np.ndarray):
    assert H.shape==(4,4)
    hom_cloud = np.ones((cloud.shape[0],4))
    hom_cloud[:, :3] = np.copy(cloud)
    transformed_cloud = H@hom_cloud.T
    return transformed_cloud[:3, :].T
