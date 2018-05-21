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
# shift the Gaussian projection by a desired amount.
#
# Considering the r direction, r_d^max, the destination r value for
# max connection strength for a given r_s is given by
#
# r_d^max = r_s + offsetd1r
#
# Thus for positive offsetd1r, "connections are stronger in the
# positive r direction away from the source".
#
# For positive offsetd0p, "connections are stronger in the
# positive p direction away from the source".

def connectionFunc(srclocs,dstlocs,sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r):

  import math
  import numpy as np
  from numba import cuda, float32, int32
  from operator import gt

  # Compute once only
  M_f_start=nfs/(E2*math.log((fovshift/(2*E2))+1))
  nfs_sq = nfs*nfs

  # Copy srclocs and dstlocs to device memory before starting
#  src_ar = cuda.to_device(np.array(srclocs, dtype=np.int32))
#  dst_ar = cuda.to_device(np.array(dstlocs, dtype=np.int32))
  src_ar = np.array(srclocs, dtype=np.int32)
  dst_ar = np.array(dstlocs, dtype=np.int32)

  # Large array of weights. Could make device only saving 3.5 s.
  weight_ar =  np.zeros((nfs_sq*nfs_sq,), dtype=np.float32)
  print ("weight_ar shape: " + str(weight_ar.shape))

  # Results array to take the non-zero weights from weight_ar.
  #
  # results is n by 4: [src_idx, dst_idx, delay, weight]
  #res_ar = cuda.to_device(np.zeros((6000000,4), dtype=np.float32))
  #res_idx = cuda.to_device(np.zeros((1,), dtype=np.int32))
  # or
  res_ar = np.zeros((6250000,4), dtype=np.float32)

  # An index to be used into the final results table, res_ar
  res_idx = np.zeros((1,), dtype=np.int32)

  # Bank-conflict avoidance means being careful with indexing
  # This version allows you to skip in groups of (3)
  @cuda.jit(device=True)
  def shifted_idx_skipping (idx):
    skip_width = 3
    num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
    bank_width_int32 = 768 # would 32 and 384 be ok? I though 32 and
                           # 768 would be ok, as this is 3kB for 32
                           # banks of shared memory.
    max_idx = (bank_width_int32 * num_banks) // skip_width  # Can't deal with a larger n than this
    #if bank_width_int32 % skip_width:
    #    return -2
    bank_width_groups = bank_width_int32 // skip_width
    idx_idiv_num_banks = idx // num_banks
    idx_mod_num_banks = idx % num_banks
    offs_idx = skip_width * ((bank_width_groups * idx_mod_num_banks) + (idx_idiv_num_banks))
    return offs_idx

  # Use @cuda.jit(device=True) to write a device function
  @cuda.jit("void(float32, int32, float32[:,:], float32[:,:], float32[:], float32,float32,float32,float32,float32,float32,int32,int32)")
  def dowork (M_f_start, nfs_sq, src_ar, dst_ar, weight_ar, sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r):
    # Work out i_src and i_dst based on the 2-D thread index
    i_src, i_dst = cuda.grid(2)
    if i_src < nfs_sq and i_dst < nfs_sq:

      # Temporary shared memory for weights
      tmp_w = cuda.shared.array(12288, dtype=float32) # Note - allocating ALL shared memory here.
      myidx = (cuda.threadIdx.x*cuda.blockDim.y + cuda.threadIdx.x)
      offsidx = shifted_idx_skipping(myidx)
      tmp_w[offsidx] = float32(0.0)
      tmp_w[offsidx+1] = float32(0.0)
      tmp_w[offsidx+2] = float32(0.0)
      cuda.syncthreads()

      # Compute the location of src_ar, this defines what sigma will be. As r (as opp. to phi) increases, the sigma should increase.
      M_f = nfs/(E2*math.log(((1+src_ar[i_src,1])/(2*E2))+1))

      # Set some of M_f to 1 to ensure the fan-out starts at around the edge of the foveal region.
      if (1+src_ar[i_src,1]) < fovshift:
        M_f = M_f_start

      # Compute modified sigma and 3 times this value
      _sigma = (sigma_m/M_f) - (sigma_m/M_f_start) + sigma_0 # as function of r, aka src_ar[1]. M_f is the function of r.
      three_sigma = 3 * _sigma

      # in-xy-plane distance (ignore src_ar[2]/dstdoc[2])
      xd = (src_ar[i_src,0] - dst_ar[i_dst,0] + offsetd0p)
      yd = (src_ar[i_src,1] - dst_ar[i_dst,1] + offsetd1r)
      if abs(xd) < three_sigma and abs(yd) < three_sigma:
        dist = math.sqrt(math.pow(xd,2) + math.pow(yd,2))
        gauss = math.exp(-0.5*math.pow(dist/_sigma,2))
        if gauss > W_cut:
          # Write result into weight_ar
          tmp_w[offsidx] = float32(gauss)
          tmp_w[offsidx+1] = float32(i_src)
          tmp_w[offsidx+2] = float32(i_dst)

      # Sync threads, then access device memory with any results
      cuda.syncthreads()
      if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        # Write data from tmp_w to res_ar, but only in ONE thread from the threadblock. Should avoid racing.
        for idx in range(0,512):
          offsidx = shifted_idx_skipping (idx)
          theweight = tmp_w[offsidx]
          #if gt(theweight, W_cut): # Testing makes no difference to speed
          # Add to weight_ar
          weight_idx = int32(tmp_w[offsidx+1])*nfs_sq + int32(tmp_w[offsidx+2])
          weight_ar[weight_idx] = theweight

      cuda.syncthreads()


  # Skipping two at a time for prescan
  @cuda.jit(device=True)
  def shifted_idx_skip2 (idx):
    skip_width = 2
    num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
    bank_width_int32 = 768 # would 32 and 384 be ok? I though 32 and
                           # 768 would be ok, as this is 3kB for 32
                           # banks of shared memory.
    max_idx = (bank_width_int32 * num_banks) // skip_width  # Can't deal with a larger n than this
    #if bank_width_int32 % skip_width:
    #    return -2
    bank_width_groups = bank_width_int32 // skip_width
    idx_idiv_num_banks = idx // num_banks
    idx_mod_num_banks = idx % num_banks
    offs_idx = skip_width * ((bank_width_groups * idx_mod_num_banks) + (idx_idiv_num_banks))
    return offs_idx

  # parallel scan for stream compaction (See sect. 39.3 https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
  @cuda.jit
  def reduceit(res_ar, weight_ar, res_idx):
    # FIXME next.
    cuda.syncthreads()

  # Now the kernel function is defined, call it
  threadsperblock = (16,32) # 8 warps to a block; 256 threads to a block. 512 might be optimal?
  blockspergrid = (1+(nfs_sq // threadsperblock[0]), 1+(nfs_sq // threadsperblock[1]))
  dowork[blockspergrid, threadsperblock](M_f_start, nfs_sq, src_ar, dst_ar, weight_ar, sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r)
  reduceit[blockspergrid*threadsperblock](weight_ar, res_ar, res_idx)


  # Now get results from device, sort, and return out. Crazy slow.
  #h_res_ar = res_ar.copy_to_host() # numpy.ndarray
  #h_res_idx = res_idx.copy_to_host()
  print("Result array shape " + str(res_ar.shape))
  print ("h_res_idx: " + str(res_idx[0]))
  #for idx in range(0,h_res_ar.shape[0]):
  #out = res_ar[0:res_idx[0]]
  out = res_ar
  return out
# end connectionFunc

#
# 2) Compute weights for the map
#

# Set up some parameters
rowlen = 50
sigma_m = 35 # was 25
E2 = 2.5
sigma_0 = 0.3
normpower = 0
fovshift = 4
W_cut = 0.001

offsetd0p = 0
offsetd1r = 0

# Containers for source/destination locations
srclocs = []
# Make srclocs - this makes rowlen x rowlen grids:
for i in range(0, rowlen):
    for j in range(0, rowlen):
        srcloc=[j,i,0] # [x,y,z] locations
        srclocs.append(srcloc)

# Call the connectionFunc to generate result
result = connectionFunc (srclocs,srclocs,sigma_m,E2,sigma_0,fovshift,rowlen,W_cut,offsetd0p,offsetd1r)
print ("Done computing")


#
# 3) Show weight results for one particular source neuron projecting
# out to destination neurons.
#
show_graphs = 0
if show_graphs>0:
    import math

    # The source neuron index to look at the connection pattern
    src_index = 325
    src_index1 = 1275
    src_index2 = 1475

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
    plt.show()
