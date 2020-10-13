#PARNAME=sigma_m        #LOC=1,1
#PARNAME=E2             #LOC=1,2
#PARNAME=sigma_0        #LOC=2,1
#PARNAME=fovshift       #LOC=2,2
#PARNAME=nfs            #LOC=3,1
#PARNAME=W_cut          #LOC=3,2
#PARNAME=offsetd0p      #LOC=4,1
#PARNAME=offsetd1r      #LOC=4,2
#HASWEIGHT

# Compute a widening Gaussian connection function for a retinotopic space, to
# maintain a constant Gaussian width in Cartesian space. This is much like the
# WideningGaussian connection function, but with a configurable neural field
# size width (W_nfs).
#
# This version incorporates an offset for dstloc[0] and dstloc[1] to shift the
# Gaussian projection by a desired amount.
#
# Considering the r direction, r_d^max, the destination r value for max
# connection strength for a given r_s is given by
#
# r_d^max = r_s + offsetd1r
#
# Thus for positive offsetd1r, "connections are stronger in the positive r
# direction away from the source".
#
# For positive offsetd0p, "connections are stronger in the positive p direction
# away from the source".

def connectionFunc(srclocs,dstlocs,sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r):
  import math
  import time # for code profiling
  time_start = int(round(time.time() * 1000))
  M_f_start=nfs/(E2*math.log((fovshift/(2*E2))+1))
  i_src = 0
  out = []
  for srcloc in srclocs:
    i_dst = 0
    # Compute the location of srcloc, this defines what sigma will be. As r (as
    # opp. to phi) increases, the sigma should increase.
    M_f =  nfs/(E2*math.log(((1+srcloc[1])/(2*E2))+1))

    # Set some of M_f to 1 to ensure the fan-out starts at around the edge of
    # the foveal region.
    if (1+srcloc[1]) < fovshift:
      M_f = M_f_start

    _sigma = (sigma_m/M_f) - (sigma_m/M_f_start) + sigma_0
    three_sigma = 3 * _sigma

    for dstloc in dstlocs:
      # in-xy-plane distance (ignore srcloc[2]/dstdoc[2])
      xd = (srcloc[0] - dstloc[0] + offsetd0p)
      yd = (srcloc[1] - dstloc[1] + offsetd1r)
      if abs(xd) < three_sigma and abs(yd) < three_sigma:
        dist = math.sqrt(math.pow(xd,2) + math.pow(yd,2))
        gauss = math.exp(-0.5*math.pow(dist/_sigma,2))
        if gauss > W_cut:
          conn = (i_src,i_dst,0,gauss)
          out.append(conn)

      i_dst = i_dst + 1
    i_src = i_src + 1

  time_donework = int(round(time.time() * 1000))
  print ("computed weights after {0} ms".format(time_donework - time_start));

  return out
# end connectionFunc
