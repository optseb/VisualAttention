import math
import numpy as np
from numba import cuda, float32, int32
from operator import gt

@cuda.jit(device=True)
def shifted_idx (idx):
    num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
    bank_width_int32 = 768
    #max_idx = (bank_width_int32 * num_banks)
    idx_idiv_num_banks = idx // num_banks
    idx_mod_num_banks = idx % num_banks
    offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks))
    return offs_idx

# A no-op version of the shifted index function
@cuda.jit(device=True)
def __shifted_idx (idx):
    return idx

# parallel scan for stream compaction (See sect. 39.3
# https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
#
# Commented out print() calls can be used if running in simulation
# mode, for which do:
#
# bash$ export NUMBA_ENABLE_CUDASIM=1
#
# Before calling python scan_test.py
#
@cuda.jit("void(float32[:], float32[:], float32[:], int32, int32)")
def reduceit(scan_ar_, weight_ar_, carry_, n, arraysz):
    thid = cuda.threadIdx.x
    tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
    d = n//2
    #print ("reduceit: thid+tb_offset:" + str(thid+tb_offset) + ", arraysz-d=" + str(arraysz-d))
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
          carry_[cuda.blockIdx.x] = temp[nm1s];
          #print("reduceit: carry_[" + str(cuda.blockIdx.x) + "] = " + str(temp[nm1s]))
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
        #print ("reduceit about to set scan_ar_[" + str(ai+tb_offset) + "]= " + str(temp[ai_s]) + ", scan_ar_[" + str(bi+tb_offset) + "]=" + str(temp[bi_s]))
        scan_ar_[ai+tb_offset] = temp[ai_s]
        scan_ar_[bi+tb_offset] = temp[bi_s]
        #print ("reduceit: scan_ar_[" + str(ai+tb_offset) + "]= " + str(scan_ar_[ai+tb_offset]) + ", scan_ar_[" + str(bi+tb_offset) + "]=" + str(scan_ar_[bi+tb_offset]))
    # End of reduceit()

# Last job is to add on the carry to each part of scan_ar WHILE AT THE SAME TIME SUMMING WITHIN A BLOCK
@cuda.jit
def sum_scans(new_carry_ar_, scan_ar_, scan_ar_sz, carry_ar_, carry_offset):
    thid = cuda.threadIdx.x
    tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
    if cuda.blockIdx.x > 0 and thid+tb_offset < scan_ar_sz:
        #print("In sum_scans: adding carry_ar_[" + str(cuda.blockIdx.x-1+carry_offset) + "]=" + str(carry_ar_[cuda.blockIdx.x-1+carry_offset]) + " to scan_ar_[" + str(thid+tb_offset) + "]=" + str(scan_ar_[thid+tb_offset]))
        new_carry_ar_[thid+tb_offset] = scan_ar_[thid+tb_offset] + carry_ar_[cuda.blockIdx.x-1+carry_offset]
        #print("              to give new_carry_ar_[" + str(thid+tb_offset) + "]=" + str(new_carry_ar_[thid+tb_offset]))
    elif cuda.blockIdx.x == 0 and thid+tb_offset < scan_ar_sz:
        new_carry_ar_[thid+tb_offset] = scan_ar_[thid+tb_offset]
        #print("In sum_scans (no add): new_carry_ar_[" + str(thid+tb_offset) + "]=" + str(new_carry_ar_[thid+tb_offset]))

#
# Build input data for the test
#

rowlen = 22500
arraysz = rowlen*rowlen

# Parameters to call reduceit
threadsperblock = 128 #4 # 128 is 1 Multiprocessor.
blockspergrid = math.ceil(arraysz/threadsperblock)

# To pad the arrays out to exact number of blocks
if arraysz%threadsperblock:
    arrayszplus = arraysz + threadsperblock - arraysz%threadsperblock
else:
    arrayszplus = arraysz

weight_ar = np.zeros((arrayszplus,), dtype=np.float32)
scan_ar = np.zeros((arrayszplus,), dtype=np.float32)

# Now some non-zero weights
weight_ar[0] = 1
weight_ar[1] = 1
weight_ar[2] = 1
weight_ar[12] = 1
weight_ar[22] = 1
weight_ar[33] = 1
weight_ar[44] = 1
weight_ar[45] = 1
weight_ar[55] = 1
weight_ar[63] = 1
weight_ar[64] = 1
weight_ar[65] = 1
weight_ar[77] = 1
weight_ar[79] = 1
weight_ar[80] = 1
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

#weight_ar[3580] = 20

# Explicitly copy to device
d_weight_ar = cuda.to_device (weight_ar)
d_scan_ar = cuda.to_device (scan_ar)

# Data structure to hold the final corrected scan
scanf = np.zeros((arrayszplus,), dtype=np.float32)
d_scanf = cuda.to_device(scanf)

# We now have the arrays in d_carrylist, which needs to be scanned, so that it
# can be added to each block of d_scan_ar. If d_carry is of large
# size, then we need to recursively scan until we do a single scan on
# the multiprocessor

#
# Make up a list of carry vectors and allocate device memory
#
carrylist = []
d_carrylist = []
# And containers for the scan
scanlist = []
d_scanlist = []
asz = arraysz
while asz > threadsperblock:

    carrysz = math.ceil (asz / threadsperblock)
    # Ensure carrysz is a multiple of threadsperblock:
    if carrysz%threadsperblock:
        carrysz = carrysz + threadsperblock - carrysz%threadsperblock

    carrylist.append (np.zeros((carrysz,), dtype=np.float32))
    d_carrylist.append (cuda.to_device(carrylist[-1]))
    asz = math.ceil (asz / threadsperblock)
    scansz = asz
    if scansz%threadsperblock:
        scansz = scansz + threadsperblock - scansz%threadsperblock
    scanlist.append (np.zeros((scansz,), dtype=np.float32))
    d_scanlist.append (cuda.to_device(scanlist[-1]))

    #print ("Allocated carry array of size " + str(carrysz))
    #print ("Allocated next scan array of size " + str(scansz))
    #print ("asz=" + str(asz))

#print ("After carrylist allocation, asz is " + str(asz) +  " and size of lists is: " + str(len(scanlist)))

# Add a last carrylist, as this will be required as a dummy carry list for the last call to reduceit()
carrylist.append (np.zeros((1,), dtype=np.float32))
d_carrylist.append (cuda.to_device(carrylist[-1]))

#
# Compute partial scans of the top-level weight_ar and the lower level
# partial sums
#
asz = arrayszplus
# The first input is the weight array, compute block-wise prefix-scan sums:
#print ("0. asz=" + str(asz) + ", scanblocks=" + str(blockspergrid) + ", scanarray length: " + str(len(scan_ar)) + ", carry(out) size: " + str(len(carrylist[0])) )
reduceit[blockspergrid, threadsperblock](d_scan_ar, d_weight_ar, d_carrylist[0], threadsperblock, asz)

asz = math.ceil (asz / threadsperblock)
j = 0
while asz > threadsperblock:
    scanblocks = math.ceil (asz / threadsperblock) # still 21. Needs to be 24.
    scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
    reduceit[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))
    asz = scanblocks
    j = j+1
# Plus one more iteration:
scanblocks = math.ceil (asz / threadsperblock)
scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
reduceit[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))

#
# Construct the scans back up the tree by summing the "carry" into the "scans"
#
ns = len(scanlist)
j = ns
#print("Before sum_scans(), j,ns=" + str(ns) + ", len(scanlist[j-1]=" + str(len(scanlist[j-1])))
while j > 0:
    sumblocks = math.ceil(len(scanlist[j-1])/threadsperblock)
    #print ("sumblocks is " + str(sumblocks) + ", length of scanlist is " + str(len(scanlist[j-1])) + ", j is " + str(j) + " d_scanlist computed is " + str(j-1))
    #                                                                          v-- becomes d_scanlist[j]?
    sum_scans[sumblocks, threadsperblock](d_carrylist[j-1], d_scanlist[j-1], len(scanlist[j-1]), d_carrylist[j], 1)
    # Now d_carrylist[j-1] has had its carrys added from the lower level
    j = j-1

# The final sum_scans() call.
#                                         out      scan                carry
sum_scans[blockspergrid, threadsperblock](d_scanf, d_scan_ar, arrayszplus, d_carrylist[0], 1)


# Copy data, device to host
r_scanf = d_scanf.copy_to_host()
r_scan_ar = d_scan_ar.copy_to_host()
r_weight_ar = d_weight_ar.copy_to_host()

j = 0
while j < 512:
    print ("weight_ar[" + str(j) + "] = " + str(r_weight_ar[j]) + " ... scan_ar[" + str(j) + "] = " + str(r_scan_ar[j]) + " ... scanf[]=" + str(r_scanf[j]))
    j = j+1

print ("threadsperblock: " + str(threadsperblock) + " blockspergrid: " + str(blockspergrid))
