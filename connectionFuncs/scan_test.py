import math
import numpy as np
from numba import cuda, float32, int32
from operator import gt

@cuda.jit(device=True)
def __shifted_idx (idx):
    num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
    bank_width_int32 = 768
    #max_idx = (bank_width_int32 * num_banks)
    idx_idiv_num_banks = idx // num_banks
    idx_mod_num_banks = idx % num_banks
    offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks))
    return offs_idx

# A no-op version of the shifted index function
@cuda.jit(device=True)
def shifted_idx (idx):
    return idx

# parallel scan for stream compaction (See sect. 39.3
# https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
#
# Commented out print() calls can be used if running in simulation
# mode, for which do:
#
# bash$ export NUMBA_ENABLE_CUDASIM=1
#
# Before calling pythong scan_test.py
#
@cuda.jit("void(float32[:], float32[:], float32[:], int32, int32)")
def reduceit(scan_ar_, weight_ar_, carry_, n, arraysz):
    thid = cuda.threadIdx.x
    tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
    d = n//2
    if (thid+tb_offset) < (arraysz-d):

        temp = cuda.shared.array(12288, dtype=float32) # Note - allocating ALL shared memory here.
        ai = thid # within one block
        bi = ai + d
        ai_s = shifted_idx(ai)
        bi_s = shifted_idx(bi)

        temp[ai_s] = weight_ar_[ai+tb_offset]
        temp[bi_s] = weight_ar_[bi+tb_offset]

        offset = 1
        # Upsweep: Ebuild sum in place up the tree
        while (d > 0):
          cuda.syncthreads()
          if thid < d:
            # Block B
            ai = offset*(2*thid+1)-1
            bi = offset*(2*thid+2)-1
            ai_s = shifted_idx(ai)
            bi_s = shifted_idx(bi)
            #if cuda.blockIdx.x == 0:
            #    print ("In upsweep, d=" + str(d) + ", temp[" + str(ai_s) + "]=" + str(temp[ai_s]) + " temp[" + str(bi_s) + "]=" + str(temp[bi_s]))
            temp[bi_s] += temp[ai_s]
            #if cuda.blockIdx.x == 0:
            #    print ("In upsweep, d=" + str(d) + ", now temp[" + str(bi_s) + "]=" + str(temp[bi_s]))

          offset *= 2
          d >>= 1

        cuda.syncthreads()

        # Block C: clear the last element - the first step of the downsweep
        if (thid == 0):
          nm1s = shifted_idx(n-1)
          # Carry last number in the block
          carry_[cuda.blockIdx.x+1] = temp[nm1s];
          # Zero it
          temp[nm1s] = 0

        # Downsweep: traverse down tree & build scan
        d = 1
        while d < n:
          offset >>= 1
          cuda.syncthreads()
          if (thid < d):
            # Block D
            ai = offset*(2*thid+1)-1
            bi = offset*(2*thid+2)-1
            ai_s = shifted_idx(ai)
            bi_s = shifted_idx(bi)
            #if cuda.blockIdx.x == 0:
            #    print ("ai_s=" + str(ai_s) + " bi_s=" + str(bi_s))
            t = temp[ai_s]
            temp[ai_s] = temp[bi_s]
            temp[bi_s] += t
            #if cuda.blockIdx.x == 0:
            #    print("After downsweep: temp[" + str(bi_s) + "]=" + str(temp[bi_s]))

          d *= 2
        cuda.syncthreads()

        # Block E: write results to device memory
        scan_ar_[ai+tb_offset] = temp[ai_s]
        scan_ar_[bi+tb_offset] = temp[bi_s]
    # End of reduceit()

# Last job is to add on the carry to each part of scan_ar
@cuda.jit
def reduceit2(scan_ar_, arraysz, carry_):
    thid = cuda.threadIdx.x
    tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
    j = 0
    allcarry = 0.0
    # This while does a scan operation, very slowly.
    while j<=cuda.blockIdx.x:
        allcarry += carry_[j]
        j += 1
    if (thid+tb_offset) < arraysz:
        scan_ar_[thid+tb_offset] += allcarry

#
# Build input data for the test
#

rowlen = 20
arraysz = rowlen*rowlen

weight_ar = np.zeros((arraysz,), dtype=np.float32)
scan_ar = np.zeros((arraysz,), dtype=np.float32)

# Now some non-zero weights
weight_ar[0] = 1
weight_ar[1] = 1
weight_ar[2] = 1
weight_ar[63] = 1
weight_ar[64] = 1
weight_ar[65] = 1
weight_ar[128] = 1
weight_ar[129] = 1
weight_ar[130] = 1
weight_ar[191] = 1
weight_ar[192] = 1
weight_ar[193] = 1
weight_ar[254] = 1
weight_ar[255] = 1
weight_ar[256] = 1
weight_ar[257] = 1


# Explicitly copy to device
d_weight_ar = cuda.to_device (weight_ar)
d_scan_ar = cuda.to_device (scan_ar)

# Parameters to call reduceit
threadsperblock = 128 # 128 is 1 Multiprocessor.
blockspergrid = 4

# Data structure to hold the carried sums
carry = np.zeros(((blockspergrid+1),), dtype=np.float32)
d_carry = cuda.to_device(carry)

reduceit[blockspergrid, threadsperblock](d_scan_ar, d_weight_ar, d_carry, threadsperblock, arraysz)
reduceit2[blockspergrid, threadsperblock](d_scan_ar, arraysz, d_carry)

# Copy device to host??
r_carry = d_carry.copy_to_host()
r_scan_ar = d_scan_ar.copy_to_host()
r_weight_ar = d_weight_ar.copy_to_host()

j = 0
while j < arraysz:
    print ("weight_ar[" + str(j) + "] = " + str(r_weight_ar[j]) + " ... scan_ar[" + str(j) + "] = " + str(r_scan_ar[j]))
    j = j+1

print ("carry:")
j = 0
while j < (blockspergrid+1):
    print ("carry[" + str(j) + "] = " + str(r_carry[j]))
    j = j+1

print ("threadsperblock: " + str(threadsperblock) + " blockspergrid: " + str(blockspergrid))
