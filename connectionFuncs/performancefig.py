import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sebcolour
col = sebcolour.Colour

# Set plotting defaults
fs = 18
fnt = {'family' : 'Arial',
       'weight' : 'regular',
       'size'   : fs}
matplotlib.rc('font', **fnt)

# IMPORTANT for svg output of text as things that can be edited in inkscape
plt.rcParams['svg.fonttype'] = 'none'

# And plot this longhand here:
F1 = plt.figure (figsize=(6,6))

# Threadbeast results Threadripper/Quadro 5000
nfs = [ 50, 75, 100, 120, 135, 150, 155, 170, 180, 192]
gpu_tb = [ 4085, 4982, 7217, 8507, 10826, 13570, 13471, 17367, 20056, 23413 ]
cpu_tb = [ 2246, 9825, 28645, 59917, 92477, 147860, 164408, 243953, 306061, 389666 ]

# Alienmonster results i9/GTX1080
cpu_am = [ 1647, 7105, 21836, 43691, 69051, 105198, 118240, 162258, 200905, 261647] # plot vs nfs

nfs_am = [ 50, 75, 100, 120, 135, 150, 155 ] # special for gpu_am
gpu_am = [ 3040, 4011, 5605, 7252, 9024, 10892, 11475 ]

# A few results for cube with an RTX 3090. Results in milliseconds,
# taken from the penultimate script line 'Computing took xxxx ms'
nfs_cube = [ 50, 150, 192, 212]
gpu_cube = [ 2348, 6197, 9062, 11001]

p=2 # power
ax2 = F1.add_subplot(2,1,1)
ax2.plot (np.power(nfs, p), np.log(np.divide(cpu_tb, 1000)), label='TR', color=col.black, linestyle='--', marker='s', markerfacecolor=col.black, markeredgecolor=col.white)
ax2.plot (np.power(nfs, p), np.log(np.divide(cpu_am, 1000)), label='i9', color=col.black, marker='v', markerfacecolor=col.black, markeredgecolor=col.white)
ax2.plot (np.power(nfs, p), np.log(np.divide(gpu_tb, 1000)), label='Quadro', color=col.dodgerblue2, linestyle='--', marker='o', markerfacecolor=col.black, markeredgecolor=col.white)
ax2.plot (np.power(nfs_am, p), np.log(np.divide(gpu_am, 1000)), label='1080', color=col.dodgerblue2, marker='^', markerfacecolor=col.black, markeredgecolor=col.white)
ax2.plot (np.power(nfs_cube, p), np.log(np.divide(gpu_cube, 1000)), label='3090', color=col.lightpink2, marker='^', markerfacecolor=col.black, markeredgecolor=col.white)
ax2.set_xlabel('$n$')
ax2.set_ylim((0,6.5))
ax2.set_ylabel('log(t)')
ax2.legend(fontsize=12)

ax3 = F1.add_subplot(2,1,2)
ax3.plot (np.power(nfs, 4), np.divide(cpu_tb,gpu_tb), label='TR/Quadro', color=col.black, linestyle='--', marker='o', markerfacecolor=col.black, markeredgecolor=col.white)
cpu_am_red = cpu_am[:-3] # miss off last 3 for this graph
ax3.plot (np.power(nfs_am, 4), np.divide(cpu_am_red,gpu_am), label='i9/1080', color=col.black, marker='^', markerfacecolor=col.black, markeredgecolor=col.white)
ax3.set_xlabel('$n$')
ax3.set_ylim((0,18))
ax3.set_ylabel('${t_{c}}/{t_{g}}$')
ax3.legend(fontsize=12)

F1.tight_layout()

plt.savefig('performance.svg')
plt.show()
