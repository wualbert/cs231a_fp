import numpy as np
from scipy.spatial.transform import Rotation

def generate_random_H(translation_lim = 1., random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    R = Rotation.random(random_state=random_state)
    rotation_matrix = R.as_matrix()
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = random_state.uniform(-translation_lim, translation_lim, 3)
    return H
