#
# 1) Define the connectionFunc - the code that will be pasted into the
# python script settings window in SpineCreator.
#

#PARNAME=sigma_m        #LOC=1,1
#PARNAME=E2             #LOC=1,2
#PARNAME=sigma_0        #LOC=2,1
#PARNAME=fovshift       #LOC=2,2
#PARNAME=nfs            #LOC=3,1
#PARNAME=W_cut          #LOC=3,2
#PARNAME=offsetd0p      #LOC=4,1
#PARNAME=offsetd1r      #LOC=4,2
#HASWEIGHT

# Compute a widening Gaussian connection function for a retinotopic
# space, to maintain a constant Gaussian width in Cartesian
# space. This is much like the WideningGaussian connection function,
# but with a configurable neural field size width (W_nfs).
#
# This version incorporates an offset for dstloc[0] and dstloc[1] to
# shift the Gaussian projection by a desired amount AND it does so to dual Gaussians, offset by +/=offsetd0p and +/-offsetd1r
#
# offsetd0p is the +/- phi direction
# offsetd1r is in the +/- r direction

def connectionFunc(srclocs,dstlocs,sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r):

  import math

  M_f_start=nfs/(E2*math.log((fovshift/(2*E2))+1))

  i_src = 0
  out = []
  for srcloc in srclocs:
    i_dst = 0
    # Compute the location of srcloc, this defines what sigma will be. As r (as opp. to phi) increases, the sigma should increase.
    M_f =  nfs/(E2*math.log(((1+srcloc[1])/(2*E2))+1))

    # Set some of M_f to 1 to ensure the fan-out starts at around the edge of the foveal region.
    if (1+srcloc[1]) < fovshift:
      M_f = M_f_start

    _sigma = (sigma_m/M_f) - (sigma_m/M_f_start) + sigma_0 # as function of r, aka srcloc[1]. M_f is the function of r.

    for dstloc in dstlocs:

      # in-xy-plane distance (ignore srcloc[2]/dstdoc[2])
      dist1 = math.sqrt(math.pow((srcloc[0] - dstloc[0] + offsetd0p),2) + math.pow((srcloc[1] - dstloc[1] + offsetd1r),2))
      dist2 = math.sqrt(math.pow((srcloc[0] - dstloc[0] - offsetd0p),2) + math.pow((srcloc[1] - dstloc[1] - offsetd1r),2))

      gauss1 = math.exp(-0.5*math.pow(dist1/_sigma,2))
      gauss2 = math.exp(-0.5*math.pow(dist2/_sigma,2))

      if gauss1 > W_cut:
        conn = (i_src,i_dst,0,gauss1)
        out.append(conn)
      elif gauss2 > W_cut:
        conn = (i_src,i_dst,0,gauss2)
        out.append(conn)

      i_dst = i_dst + 1
    i_src = i_src + 1
  return out
# end connectionFunc

#
# 2) Compute weights for the map
#

# Set up some parameters
rowlen = 50
sigma_m = 25
E2 = 2.5
sigma_0 = 0.3
normpower = 0
fovshift = 4
nfs = 50
W_cut = 0.001

offsetd0p = 0
offsetd1r = 6

# Containers for source/destination locations
srclocs = []
# Make srclocs - this makes rowlen x rowlen grids:
for i in range(0, rowlen):
    for j in range(0, rowlen):
        srcloc=[j,i,0] # [x,y,z] locations
        srclocs.append(srcloc)

# Call the connectionFunc to generate result
result = connectionFunc (srclocs,srclocs,sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r)

#offsetd0p = 6
#offsetd1r = 0
#result2 = connectionFunc (srclocs,srclocs,sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r)
print "Done computing"


#
# 3) Show weight results for one particular source neuron projecting
# out to destination neurons.
#

import math

# The source neuron index to look at the connection pattern
src_index = 25
src_index1 = 1275
src_index2 = 2475

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

twofigs = 0
if twofigs:
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

    for res in result2:
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
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(xs,ys,ws)
    ax2.scatter(xs1,ys1,ws1)
    ax2.scatter(xs2,ys2,ws2)

# At end show plot
plt.show()
