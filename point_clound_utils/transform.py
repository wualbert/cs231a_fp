import numpy as np

def transform_cloud(cloud, H:np.ndarray):
    assert H.shape==(4,4)
    hom_cloud = np.ones((cloud.shape[0],4))
    hom_cloud[:, :3] = np.copy(cloud)
    transformed_cloud = H@hom_cloud.T
    return transformed_cloud[:3, :].T

def apply_gaussian_noise(cloud, sigma = np.ones(3)*1e-3,
                         rand = np.random.RandomState(0)):
    new_cloud = np.copy(cloud)
    for i in range(3):
        noise = rand.normal(0, sigma[i], cloud.shape[0])
        new_cloud[:, i]+=noise
    return new_cloud