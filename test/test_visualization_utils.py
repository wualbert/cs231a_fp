import matplotlib.pyplot as plt
import visualization_utils as viz

import numpy as np

def test_plot_error_with_min_max():
    x = np.arange(10)
    y = 2*x
    yerr = np.full(10, 2)
    ymin = 1*x
    ymax = 3*x
    fig, ax = viz.plot_error_with_min_max(x, y, yerr, ymin, ymax)
    ax.legend()
    ax.set_xlabel('Gaussian distribution $\sigma$')
    ax.set_ylabel('ADD(cm)')
    plt.show()

if __name__ == '__main__':
    test_plot_error_with_min_max()
