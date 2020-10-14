# Compute a widening Gaussian connection function for a retinotopic space
# srclocs, dstlocs: coordinates of source and destination neurons
# sigma_m: At mag. factor 1, this is the width of the Gaussian proj.
# E2: Widening parameter     sigma_0: An offset for the proj. width
# fshift: Foveal shift       nfs: Side length of 'square' neural popn.
# W_cut: The weight cutoff   osd0p, osd1r: phi and r direction offsets
def connectionFunc(srclocs,dstlocs,sigma_m,E2,sigma_0,fshift,nfs,W_cut,osd0p,osd1r):
  import math
  M_f_start=nfs/(E2*math.log((fshift/(2*E2))+1))
  i_src = 0
  out = []
  for srcloc in srclocs:
    i_dst = 0
    # Compute the magnification factor, which varies with srcloc[1] (r)
    M_f =  nfs/(E2*math.log(((1+srcloc[1])/(2*E2))+1))
    # In the foveal region (small srcloc[1]) fix M_f to M_f_start:
    if (1+srcloc[1]) < fshift:
      M_f = M_f_start
    # Use M_f to compute the effective width of the projection:
    _sigma = (sigma_m/M_f) - (sigma_m/M_f_start) + sigma_0
    three_sigma = 3 * _sigma
    # The inner-loop, over destination neurons:
    for dstloc in dstlocs:
      # in-xy-plane distance (ignore srcloc[2]/dstdoc[2])
      xd = (srcloc[0] - dstloc[0] + osd0p)
      yd = (srcloc[1] - dstloc[1] + osd1r)
      if abs(xd) < three_sigma and abs(yd) < three_sigma:
        dist = math.sqrt(math.pow(xd,2) + math.pow(yd,2))
        gauss = math.exp(-0.5*math.pow(dist/_sigma,2))
        if gauss > W_cut:
          conn = (i_src,i_dst,0,gauss)
          out.append(conn)

      i_dst = i_dst + 1
    i_src = i_src + 1
  return out
# end connectionFunc
