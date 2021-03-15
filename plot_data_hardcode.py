import visualization_utils as vizu
import matplotlib.pyplot as plt
import numpy as np

# Drill data
obj = 'Drill'
noise = 0.0
if obj == 'Drill':
    if noise == 0:
        avg = [0.02058744,0.01387917,0.01887471,0.01570151]
        std = [0.01038669,0.00456,0.01070552,0.01098901]
    elif noise == 0.05:
        avg = [0.02837931,0.02623544,0.02550846,0.02306831]
        std = [0.03321491,0.02728322,0.02450251,0.02151715]

if obj == 'Can':
    if noise == 0:
        avg = [0.08033147,0.08197393,0.06981859,0.06568135]
        std = [0.14792633,0.13385311,0.08706189,0.11091698]
    elif noise == 0.05:
        avg = [0.10382205,0.11551697,0.09680962,0.11459004]
        std = [0.15560281,0.16115139,0.10916843,0.13545707]

colors = ['r','g','b','y']
legends = ['Segpose', 'Segpose+ICP','Segpose+Kalman','Segpose+Kalman+ICP']
fig, ax = plt.subplots()
for i in range(len(avg)):
    fig, ax = vizu.plot_error([i], avg[i], std[i], fig=fig, ax=ax,
                              color=colors[i])
ax.set_ylabel('Error')
ax.set_ylim(bottom=0)#, top=0.07)
# ax.legend(legends,loc='upper right')
ax.legend(legends,loc='lower left')
ax.plot(np.linspace(-0.25, 3.25,100), np.ones(100)*avg[0], 'r--', lw=0.5)
ax.set_title(obj+f' Pose Estimation with Noise Amount {noise}')
plt.savefig('error_plot_'+obj+f'_{noise}.png', dpi=300)