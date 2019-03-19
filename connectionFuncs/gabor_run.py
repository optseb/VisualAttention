# Import the connectionFunc
import gabor as cf

# Set up some parameters
rowlen = 150
sigma_g = 4
gain_g = 0.1
lambda_s = 16
gain_s = 1
dir_s = 145
W_cut = 0.001
max_n_weights = 20000000

# Containers for source/destination locations
srclocs = []
# Make srclocs - this makes rowlen x rowlen grids:
for i in range(0, rowlen):
    for j in range(0, rowlen):
        srcloc=[j,i,0] # [x,y,z] locations
        srclocs.append(srcloc)

# Call the connectionFunc to generate result
result = cf.connectionFunc (srclocs,srclocs,sigma_g,gain_g,lambda_s,gain_s,dir_s,W_cut,20)
#        def connectionFunc(srclocs,dstlocs,sigma_g,gain_g,lambda_s,gain_s,dir_s,wco,roi)

print ("Done computing, result is a {0}".format(type(result)))

# Show weight results for one particular source neuron projecting
# out to destination neurons.
show_graphs = 1
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
