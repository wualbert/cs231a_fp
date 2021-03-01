import numpy as np
from scipy.spatial.transform import Rotation

def generate_random_H(lim = 1., random_state=None):
    R = Rotation.random(random_state=random_state)
    rotation_matrix = R.as_matrix()
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = random_state.uniform(-lim, lim, 3)
    return H
