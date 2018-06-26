import math
import numpy as np
from numba import cuda, float32, int32
from operator import gt

@cuda.jit(device=True)
def __shifted_idx (idx):
    num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
    bank_width_int32 = 768
    # max_idx = (bank_width_int32 * num_banks)
    idx_idiv_num_banks = idx // num_banks
    idx_mod_num_banks = idx % num_banks
    offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks))
    return offs_idx

# A no-op version of the shifted index function
@cuda.jit(device=True)
def shifted_idx (idx):
    return idx

# Prenormalise, so that the scan algo works. That means any weight
# which is >0 should be set to 1, so that when the scan adds up the
# values, we get numbers that can be used as memory
# addresses. Hacky. Would prefer not to need this prenorm
# kernel. However, the complexity with the reduceit kernel is that it
# works on two parts of the weight array per thread and so my naive
# attempt to make the "summing for memory access" thing work
# failed. This will be quite fast - probably only about the cost of
# the kernel invocation, in any case. I could also possibly allocate
# another region of device RAM to hold integers, so that the memory
# addresses would be safe.
@cuda.jit
def prenorm(weight_ar_, nonzero_ar_, arraysz):
    thid = cuda.threadIdx.x + (cuda.blockIdx.x*cuda.blockDim.x)
    if thid < arraysz:
        nonzero_ar_[thid] = 1 if weight_ar_[thid] > 0.0 else 0

# parallel scan for stream compaction (See sect. 39.3
# https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
#
# Commented out print() calls can be used if running in simulation
# mode, for which do:
#
# bash$ export NUMBA_ENABLE_CUDASIM=1
#
# Before calling python scan2_recursive.py
#
@cuda.jit
def reduceit(scan_ar_, nonzero_ar_, carry_, n, arraysz):
    thid = cuda.threadIdx.x
    tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
    d = n//2

    #print ("reduceit: if (thid+tb_offset:" + str(thid+tb_offset) + " < arraysz-d=" + str(arraysz-d) + ")")
    # This runs for every element in nonzero_ar_
    if (thid+tb_offset) < (arraysz-d):

        temp = cuda.shared.array(12288, dtype=float32) # Note - allocating ALL shared memory here.
        ai = thid # within one block
        bi = ai + d
        ai_s = shifted_idx(ai)
        bi_s = shifted_idx(bi)

        # Summing scheme
        temp[ai_s] = nonzero_ar_[ai+tb_offset]
        temp[bi_s] = nonzero_ar_[bi+tb_offset]

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
                #    print ("In upsweep, d=" + str(d) + ", temp[bi_s:" + str(bi_s) + "]=" + str(temp[bi_s]) + " += temp[ai_s:" + str(ai_s) + "]=" + str(temp[ai_s]))
                temp[bi_s] += temp[ai_s]

                #if cuda.blockIdx.x == 0:
                #    print ("In upsweep, d=" + str(d) + ", now temp[bi_s:" + str(bi_s) + "]=" + str(temp[bi_s]))

            offset *= 2
            d >>= 1

        cuda.syncthreads()

        # Block C: clear the last element - the first step of the downsweep
        if (thid == 0):
            nm1s = shifted_idx(n-1)
            # Carry last number in the block
            carry_[cuda.blockIdx.x] = temp[nm1s];
            #print("reduceit: clear last element; carry_[" + str(cuda.blockIdx.x) + "] = " + str(temp[nm1s]))
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
        print ("reduceit about to set scan_ar_[ai=" + str(ai) + "+tb_offset=" + str(tb_offset) + " =" + str(ai+tb_offset) + "] to " + str(temp[ai_s]) + "; scan_ar_[bi=" + str(bi) + "+tboffset =" + str(bi+tb_offset) + "] to " + str(temp[bi_s]))
        scan_ar_[ai+tb_offset] = temp[ai_s]
        scan_ar_[bi+tb_offset] = temp[bi_s]
        print ("reduceit: scan_ar_[" + str(ai+tb_offset) + "]= " + str(scan_ar_[ai+tb_offset]) + ", scan_ar_[" + str(bi+tb_offset) + "]=" + str(scan_ar_[bi+tb_offset]))
    cuda.syncthreads()
    # End of reduceit()

# Last job is to add on the carry to each part of scan_ar WHILE AT THE SAME TIME SUMMING WITHIN A BLOCK
@cuda.jit
def sum_scans(new_carry_ar_, scan_ar_, scan_ar_sz, carry_ar_):
    thid = cuda.threadIdx.x
    tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset

    if cuda.blockIdx.x > 0 and thid+tb_offset < scan_ar_sz:
        #print("In sum_scans: adding carry_ar_[blockIdx.x=" + str(cuda.blockIdx.x) + "]=" + str(carry_ar_[cuda.blockIdx.x]) + " to scan_ar_[" + str(thid+tb_offset) + "]=" + str(scan_ar_[thid+tb_offset]) + " (size:" + str (scan_ar_sz) + ")")
        new_carry_ar_[thid+tb_offset] = scan_ar_[thid+tb_offset] + carry_ar_[cuda.blockIdx.x]
        #print("              to give new_carry_ar_[" + str(thid+tb_offset) + "]=" + str(new_carry_ar_[thid+tb_offset]))
    elif cuda.blockIdx.x == 0 and thid+tb_offset < scan_ar_sz:
        # This is the first block, so there's no carrying to be done; the new carry array should just contain the existing scan array.
        new_carry_ar_[thid+tb_offset] = scan_ar_[thid+tb_offset]
        #print("In sum_scans (no add): new_carry_ar_[" + str(thid+tb_offset) + "] = scan_ar_[" + str(thid+tb_offset) + "] = " + str(new_carry_ar_[thid+tb_offset]))

    cuda.syncthreads()
#
# Build input data for the test
#

rowlen = 5
arraysz = rowlen*rowlen

# Parameters to call reduceit
threadsperblock = 4 # 128 is 1 Multiprocessor.
blockspergrid = math.ceil(arraysz/threadsperblock)

# To pad the arrays out to exact number of blocks
if arraysz%threadsperblock:
    arrayszplus = arraysz + threadsperblock - arraysz%threadsperblock
else:
    arrayszplus = arraysz

# weight_ar is the input
weight_ar = np.zeros((arrayszplus,), dtype=np.float32)
# nonzero_ar is set to 1 for every element for which weight_ar is >0
nonzero_ar = np.zeros((arrayszplus,), dtype=np.uint32)
# scan_ar is going to hold the result of scanning the input
scan_ar = np.zeros((arrayszplus,), dtype=np.uint32)

# Now some non-zero, non-unary weights
weight_ar[0] = 1.1
weight_ar[1] = 1.3
weight_ar[2] = 1.2
weight_ar[3] = 0.2
weight_ar[4] = 0.3
weight_ar[12] = 1.7
weight_ar[22] = 1.9

if rowlen == 18:
    weight_ar[33] = 2.3
    weight_ar[44] = 2.3
    weight_ar[45] = 2.3
    weight_ar[55] = 2.3
    weight_ar[63] = 2.3
    weight_ar[64] = 2.3
    weight_ar[65] = 2.3
    weight_ar[77] = 2.3
    weight_ar[79] = 2.3
    weight_ar[80] = 2.3
    weight_ar[128] = 2.3
    weight_ar[129] = 2.3
    weight_ar[130] = 2.3
    weight_ar[191] = 2.3
    weight_ar[192] = 2.3
    weight_ar[193] = 2.3
    weight_ar[254] = 2.3
    weight_ar[255] = 2.3
    weight_ar[256] = 2.3
    weight_ar[257] = 2.3

# Explicitly copy to device
d_weight_ar = cuda.to_device (weight_ar)
d_nonzero_ar = cuda.to_device (nonzero_ar)
d_scan_ar = cuda.to_device (scan_ar)

# scanf_ar is a data structure to hold the final, corrected scan
scanf_ar = np.zeros((arrayszplus,), dtype=np.uint32)
d_scanf_ar = cuda.to_device(scanf_ar)

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

    print ("Allocated carry array of size " + str(carrysz))
    print ("Allocated next scan array of size " + str(scansz))
    print ("asz=" + str(asz))

print ("After carrylist allocation, asz is " + str(asz) +  " and size of lists is: " + str(len(scanlist)))

# Add a last carrylist, as this will be required as a dummy carry list for the last call to reduceit()
carrylist.append (np.zeros((1,), dtype=np.int32))
d_carrylist.append (cuda.to_device(carrylist[-1]))

#
# Compute partial scans of the top-level weight_ar and the lower level
# partial sums
#
asz = arrayszplus
# The first input is the weight array, compute block-wise prefix-scan sums:
print ("0. asz reset to " + str(asz) + ", scanblocks=" + str(blockspergrid) + ", scanarray length: " + str(len(scan_ar)) + ", carry(out) size: " + str(len(carrylist[0])) )
prenorm[blockspergrid, threadsperblock](d_weight_ar, d_nonzero_ar, asz)
reduceit[blockspergrid, threadsperblock](d_scan_ar, d_nonzero_ar, d_carrylist[0], threadsperblock, asz)

r_scan_ar = d_scan_ar.copy_to_host()
r_scanf_ar = d_scanf_ar.copy_to_host()
r_weight_ar = d_weight_ar.copy_to_host()
r_nonzero_ar = d_nonzero_ar.copy_to_host()
print ("Here comes weight_ar etc")
j = 0
while j < arraysz:
    print ("weight_ar[" + str(j) + "] = " + str(r_weight_ar[j]) + "("+str(r_nonzero_ar[j])+") ... scan_ar[" + str(j) + "] = " + str(r_scan_ar[j]) + " ... scanf_ar[]=" + str(r_scanf_ar[j]))
    j = j+1


print("AFTER FIRST reduceit BEFORE sum_scans, lists are:")
# Copy scan lists to host
j = 0
r_scanlist = []
while j < len(scanlist):
    r_scanlist.append (d_scanlist[j].copy_to_host())
    j = j+1

j = 0
r_carrylist = []
while j < len(carrylist):
    r_carrylist.append (d_carrylist[j].copy_to_host())
    j = j+1

# Print scan lists
j = 0
while j < len(r_scanlist):
    k = 0
    while k < len(r_scanlist[j]):
        print ("r_scanlist[j"+str(j)+"][k"+str(k)+"] = " + str(r_scanlist[j][k]))
        k = k+1
    j = j+1

j = 0
while j < len(r_carrylist):
    k = 0
    while k < len(r_carrylist[j]):
        print ("j: "+ str(j)+" k: "+ str(k))
        print ("r_carrylist[j"+str(j)+"][k"+str(k)+"] = " + str(r_carrylist[j][k]))
        k = k+1
    j = j+1


print ("SECOND reduceit() call-set")
asz = math.ceil (asz / threadsperblock)
j = 0
while asz > threadsperblock:
    scanblocks = math.ceil (asz / threadsperblock)
    scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
    reduceit[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))
    asz = scanblocks
    j = j+1
# Plus one more iteration:
print ("Last reduceit call, j=" + str(j))
scanblocks = math.ceil (asz / threadsperblock)
scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
reduceit[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))

print("AFTER SECOND reduceit BEFORE sum_scans, lists are:")
# Copy scan lists to host
j = 0
r_scanlist = []
while j < len(scanlist):
    r_scanlist.append (d_scanlist[j].copy_to_host())
    j = j+1

j = 0
r_carrylist = []
while j < len(carrylist):
    r_carrylist.append (d_carrylist[j].copy_to_host())
    j = j+1

# Print scan lists
j = 0
while j < len(r_scanlist):
    k = 0
    while k < len(r_scanlist[j]):
        print ("r_scanlist[j"+str(j)+"][k"+str(k)+"] = " + str(r_scanlist[j][k]))
        k = k+1
    j = j+1

j = 0
while j < len(r_carrylist):
    k = 0
    while k < len(r_carrylist[j]):
        print ("j: "+ str(j)+" k: "+ str(k))
        print ("r_carrylist[j"+str(j)+"][k"+str(k)+"] = " + str(r_carrylist[j][k]))
        k = k+1
    j = j+1


#
# Construct the scans back up the tree by summing the "carry" into the "scans"
#
ns = len(scanlist)
j = ns
print("Before sum_scans(), j,ns=" + str(ns) + ", len(scanlist[j-1]=" + str(len(scanlist[j-1])))
while j > 0:
    sumblocks = math.ceil(len(scanlist[j-1])/threadsperblock)
    print ("j=" +str(j)+") sumblocks is " + str(sumblocks))
    print ("               length of scanlist[j-1="+str(j-1)+"] is " + str(len(scanlist[j-1])))
    print ("               length of carrylist[j="+str(j)+"] is " + str(len(carrylist[j])))
    # Why is sumblocks different in different loops?
    sum_scans[sumblocks, threadsperblock](d_carrylist[j-1], d_scanlist[j-1], len(scanlist[j-1]), d_carrylist[j])
    # Now d_carrylist[j-1] has had its carrys added from the lower level
    j = j-1

# The final sum_scans() call.
#                                         out         scan                    carry
print ("Final sum_scans() call; writing into final result in d_scanf_ar blockspergrid: " + str(blockspergrid))
sum_scans[blockspergrid, threadsperblock](d_scanf_ar, d_scan_ar, arrayszplus, d_carrylist[0])


# Copy data, device to host
r_scan_ar = d_scan_ar.copy_to_host()
r_scanf_ar = d_scanf_ar.copy_to_host()
r_weight_ar = d_weight_ar.copy_to_host()
r_nonzero_ar = d_nonzero_ar.copy_to_host()

# Copy scan lists to host
j = 0
r_scanlist = []
while j < len(scanlist):
    r_scanlist.append (d_scanlist[j].copy_to_host())
    j = j+1

j = 0
r_carrylist = []
while j < len(carrylist):
    r_carrylist.append (d_carrylist[j].copy_to_host())
    j = j+1

# Print scan lists
j = 0
while j < arraysz:
    print ("weight_ar[" + str(j) + "] = " + str(r_weight_ar[j]) + "("+str(r_nonzero_ar[j])+") ... scan_ar[" + str(j) + "] = " + str(r_scan_ar[j]) + " ... scanf_ar[]=" + str(r_scanf_ar[j]))
    j = j+1

j = 0
while j < len(r_scanlist):
    k = 0
    while k < len(r_scanlist[j]):
        print ("r_scanlist[j"+str(j)+"][k"+str(k)+"] = " + str(r_scanlist[j][k]))
        k = k+1
    j = j+1

j = 0
while j < len(r_carrylist):
    k = 0
    while k < len(r_carrylist[j]):
        print ("j: "+ str(j)+" k: "+ str(k))
        print ("r_carrylist[j"+str(j)+"][k"+str(k)+"] = " + str(r_carrylist[j][k]))
        k = k+1
    j = j+1

print ("threadsperblock: " + str(threadsperblock) + " blockspergrid: " + str(blockspergrid))
