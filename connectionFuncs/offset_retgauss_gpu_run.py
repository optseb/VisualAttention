# Import the GPU connectionFunc
import offset_retgauss_gpu as org
import time

# Set up some parameters
rowlen = 180
sigma_m = 100
E2 = 2.5
sigma_0 = 0.3
fovshift = 4
W_cut = 0.01
offsetd0p = 0
offsetd1r = 0
max_n_weights = 20000000

# Containers for source/destination locations
srclocs = []
# Make srclocs - this makes rowlen x rowlen grids:
for i in range(0, rowlen):
    for j in range(0, rowlen):
        srcloc=[j,i,0] # [x,y,z] locations
        srclocs.append(srcloc)

# Call the connectionFunc to generate result
t_start = int(round(time.time() * 1000))
result = org.connectionFunc (srclocs,srclocs,sigma_m,E2,sigma_0,fovshift,rowlen,W_cut,offsetd0p,offsetd1r,max_n_weights)
t_end = int(round(time.time() * 1000))
print ("Computing took {0} ms".format(t_end-t_start))
print ("Done computing, result is a {0}".format(type(result)))

# Show weight results for one particular source neuron projecting
# out to destination neurons.
show_graphs = 0
if show_graphs>0:
    import math

    # The source neuron index to look at the connection pattern
    src_index = int(rowlen/10) * rowlen + int(rowlen/2)
    src_index1 = int(rowlen/2) * rowlen + int(rowlen/2)
    src_index2 = int(rowlen/10) * 9 * rowlen - int(rowlen/2)

    # Extract xs, ys and weights for source-to-destination connection into
    # these lists:
    xs = []
    ys = []
    ws = []
    xs1 = []
    ys1 = []
    ws1 = []
    xs2 = []
    ys2 = []
    ws2 = []

    for res in result:
        #print ('res: {0}'.format(res))
        if (res[0] == src_index):
            # In my example, the position x and y come from the
            # destination index, which is found in res[1]. res[2] contains
            # the delay (unused here). res[3] contains the weight.
            xs.append(res[1]%rowlen)
            ys.append(math.floor(res[1]/rowlen))
            ws.append(res[3])
            #print ('Appended ', res[1]%rowlen, math.floor(res[1]/rowlen), res[3])
        elif (res[0] == src_index1):
            xs1.append(res[1]%rowlen)
            ys1.append(math.floor(res[1]/rowlen))
            ws1.append(res[3])
        elif (res[0] == src_index2):
            xs2.append(res[1]%rowlen)
            ys2.append(math.floor(res[1]/rowlen))
            ws2.append(res[3])

    # Now do a scatter plot of the weights
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs,ys,ws)
    ax.scatter(xs1,ys1,ws1)
    ax.scatter(xs2,ys2,ws2)
    ax.set_xlim([0,rowlen])
    ax.set_ylim([0,rowlen])

    plt.show()
