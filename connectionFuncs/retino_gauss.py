#
# 1) Define the connectionFunc - the code that will be pasted into the
# python script settings window in SpineCreator.
#

#PARNAME=sigma_m #LOC=1,1
#PARNAME=E2 #LOC=1,2
#PARNAME=sigma_0 #LOC=2,1
#PARNAME=normpower #LOC=2,2
#PARNAME=fovshift #LOC=3,1
#PARNAME=W_nfs #LOC=3,2
#HASWEIGHT

# Compute a widening Gaussian connection function for a retinotopic
# space, to maintain a constant Gaussian width in Cartesian
# space. This is much like the WideningGaussian connection function,
# but with a configurable neural field size width (W_nfs).

def connectionFunc(srclocs,dstlocs,sigma_m,E2,sigma_0,normpower,fovshift,W_nfs):

  import math

  M_f_start=W_nfs/(E2*math.log((fovshift/(2*E2))+1))

  i_src = 0
  out = []
  for srcloc in srclocs:
    i_dst = 0
    # Compute the location of srcloc, this defines what sigma will be. As r (as opp. to phi) increases, the sigma should increase.
    M_f =  W_nfs/(E2*math.log(((1+srcloc[1])/(2*E2))+1))

    # Set some of M_f to 1 to ensure the fan-out starts at around the edge of the foveal region.
    if (1+srcloc[1]) < fovshift:
      M_f = M_f_start
      #print 'Start M_f', M_f

    _sigma = (sigma_m/M_f) - (sigma_m/M_f_start) + sigma_0 # as function of r, aka srcloc[1]
    three_sigma = _sigma * 3

    normterm = 1/math.pow(_sigma,normpower)

    for dstloc in dstlocs:

      xd = srcloc[0] - dstloc[0]
      yd = srcloc[1] - dstloc[1]
      if abs(xd) < three_sigma and abs(yd) < three_sigma:
        dist = math.sqrt(math.pow(xd,2) + math.pow(yd,2)) #'+ math.pow((srcloc[2] - dstloc[2]),2))

        gauss = normterm*math.exp(-0.5*math.pow(dist/_sigma,2))

        if gauss > 0.001:
          #sys.stdout.write('gauss>0.0001: i={0} gauss={1}'.format( i, gauss))
          conn = (i_src,i_dst,0,gauss)
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
W_nfs = 50

# Containers for source/destination locations
srclocs = []
# Make srclocs - this makes rowlen x rowlen grids:
for i in range(0, rowlen):
    for j in range(0, rowlen):
        srcloc=[j,i,0] # [x,y,z] locations
        srclocs.append(srcloc)

# Call the connectionFunc to generate result
result = connectionFunc (srclocs, srclocs, sigma_m, E2, sigma_0, normpower, fovshift, W_nfs)
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
#plt.show()
