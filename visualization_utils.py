import matplotlib.pyplot as plt

def plot_error_with_min_max(x, y, yerr, ymin, ymax,
                            fig = None, ax=None, color='k',
                            label='foo'):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr, fmt=f'o{color}', lw=3, label=label)
    ax.errorbar(x, y, [y - ymin, ymax - ymin],
                 fmt=f'.{color}', ecolor=f'{color}', lw=1)
    return fig, ax

def plot_error(x, y, yerr,
                fig = None, ax=None, color='k',
                label='foo'):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    if label is not None:
        ax.errorbar(x, y, yerr, fmt=f'o{color}', lw=1, label=label)
    else:
        ax.errorbar(x, y, yerr, fmt=f'o{color}', lw=1)
    return fig, ax