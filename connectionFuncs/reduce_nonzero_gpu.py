###############################################################################
import numpy as np
from numba import cuda, float32, int32

# Like prefixscan_gpu, but returning the non-zero values from
# weight_ar in d_result_idx and d_result_val.
#
# d_weight_ar - A memory area on the GPU memory containing a sparse
# matrix of results (floating point, probably)
#
# arraysz - The length of the sparse matrix contained in d_weight_ar
#
# arrayszplus - The size of the memory area d_weight_ar. This should
# be an integer multiple of threadsperblock
#
# threadsperblock - how many threads to launch per threadblock on the
# GPU.
#
# d_result_idx - a memory array to hold the result indices - the index
# into d_weight_ar that held the non-zero result.
#
# d_result_val - a memory array to hold the result values - the
# nonzero members of d_weight_ar
#
# res_sz - The size of d_result_idx and d_result_val. Unused (make
# sure your d_result* arrays have enough elements to hold all the
# results you're computing)
#
def reduce_nonzero_gpu (d_weight_ar, arraysz, arrayszplus, threadsperblock, d_result_idx, d_result_val, res_sz):
    import math
    from operator import gt

    # Shifts index to avoid bank conflicts (on GTX1070)
    @cuda.jit(device=True)
    def shifted_idx (idx):
        num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
        bank_width_int32 = 768
        # max_idx = (bank_width_int32 * num_banks)
        idx_idiv_num_banks = idx // num_banks
        idx_mod_num_banks = idx % num_banks
        offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks))
        return offs_idx

    # Detect non-zero values in weight_ar_. Update each element of the
    # identically shaped nonzero_ar_ to hold 1 for non-zero values in
    # weight_ar_ and 0 for zero values in weight_ar_
    @cuda.jit
    def detect_nonzero (weight_ar_, nonzero_ar_, arraysz):
        thid = cuda.threadIdx.x + (cuda.blockIdx.x*cuda.blockDim.x)
        if thid < arraysz:
            nonzero_ar_[thid] = 1 if weight_ar_[thid] > 0.0 else 0

    #
    # parallel prefix scan for stream compaction (See sect. 39.3
    # https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
    #
    # This sums up the values in nonzero_ar_, placing the running
    # total in scan_ar_. The final carry value is placed in carry_
    #
    # This kernel carries out the prefix scan for a single threadblock.
    #
    # scan_ar_ - The result of the prefix scan. Array of uint32s
    # (could be float32s)
    #
    # input_ar_ - The input for the algorithm. Array of uint32s (could
    # be float32s)
    #
    # carry_ - The carry array - carry_[cuda.blockIdx.x] is updated
    # with the final value in scan_ar_
    #
    # threadsperblock_ - The number of CUDA threads per threadblock
    #
    # inputsz - The size of the arrays scan_ar_ and input_ar_
    #
    @cuda.jit()
    def one_block_scan (scan_ar_, input_ar_, carry_, threadsperblock_, inputsz):
        thid = cuda.threadIdx.x                     # Thread ID
        tb_offset = cuda.blockIdx.x*cuda.blockDim.x # Threadblock offset
        d = threadsperblock_//2                     # d starts out as half the block

        # This runs for every element in input_ar_
        if (thid+tb_offset) < (inputsz-d):

            # Allocate ALL shared memory here. Use float32 as type; could be uint32.
            shmem = cuda.shared.array(12288, dtype=float32)
            ai = thid # within one block
            bi = ai + d # ai and bi are well separated across the shared memory. bi = ai+1 could also work?

            # Compute shifted indices for efficient use of shared memory
            ai_s = shifted_idx(ai)
            bi_s = shifted_idx(bi)

            # Copy input into local shared memory array
            shmem[ai_s] = input_ar_[ai+tb_offset]
            shmem[bi_s] = input_ar_[bi+tb_offset]

            offset = 1
            # Upsweep: Build sum in place up the tree
            while (d > 0):
                cuda.syncthreads()
                if thid < d:
                    # Block B
                    ai = offset*(2*thid+1)-1
                    bi = offset*(2*thid+2)-1
                    ai_s = shifted_idx(ai)
                    bi_s = shifted_idx(bi)
                    shmem[bi_s] += shmem[ai_s]

                offset *= 2
                d >>= 1

            cuda.syncthreads()

            # Block C: clear the last element - the first step of the downsweep
            if (thid == 0):
                nm1s = shifted_idx(threadsperblock-1)
                # Carry last number in the block
                carry_[cuda.blockIdx.x] = shmem[nm1s];
                shmem[nm1s] = 0

            # Downsweep: traverse down tree & build scan
            d = 1
            while d < threadsperblock_:
                offset >>= 1
                cuda.syncthreads()
                if (thid < d):
                    # Block D
                    ai = offset*(2*thid+1)-1
                    bi = offset*(2*thid+2)-1
                    ai_s = shifted_idx(ai)
                    bi_s = shifted_idx(bi)
                    t = shmem[ai_s]
                    shmem[ai_s] = shmem[bi_s]
                    shmem[bi_s] += t
                d *= 2
            cuda.syncthreads()

            # Block E: write results to device memory
            scan_ar_[ai+tb_offset] = shmem[ai_s]
            if bi < threadsperblock_:
                scan_ar_[bi+tb_offset] = shmem[bi_s]

        return # End of one_block_scan()

    # Last job is to add on the carry to each part of scan_ar WHILE AT THE SAME TIME SUMMING WITHIN A BLOCK
    @cuda.jit
    def sum_scans(new_carry_ar_, scan_ar_, scan_ar_sz, carry_ar_):
        thid = cuda.threadIdx.x
        tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
        arr_addr = thid+tb_offset

        # Try replacing with ternarys at some point:
        if cuda.blockIdx.x > 0 and arr_addr < scan_ar_sz:
            new_carry_ar_[arr_addr] = scan_ar_[arr_addr] + carry_ar_[cuda.blockIdx.x]
        elif cuda.blockIdx.x == 0 and arr_addr < scan_ar_sz:
            new_carry_ar_[arr_addr] = scan_ar_[arr_addr]

        cuda.syncthreads()

    @cuda.jit
    def sum_scans_destructively(scan_ar_, scan_ar_sz, carry_ar_):
        thid = cuda.threadIdx.x
        tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
        arr_addr = thid+tb_offset

        if cuda.blockIdx.x > 0 and arr_addr < scan_ar_sz:
            scan_ar_[arr_addr] = scan_ar_[arr_addr] + carry_ar_[cuda.blockIdx.x]

    # Use the scan array in d_scan_ar to compute the
    @cuda.jit
    def extract_nonzero (d_weight_ar, weight_sz, d_scan_ar, d_result_idx, d_result_val):
        thid = cuda.threadIdx.x + (cuda.blockIdx.x*cuda.blockDim.x)
        if thid < weight_sz and d_weight_ar[thid] > 0.0:
            d_result_idx[d_scan_ar[thid]] = thid
            d_result_val[d_scan_ar[thid]] = d_weight_ar[thid]

    #
    # Build input data for the test
    #
    arraysz = rowlen*rowlen

    blockspergrid = math.ceil(arraysz/threadsperblock)

    # To pad the arrays out to exact number of blocks
    if arraysz%threadsperblock:
        arrayszplus = arraysz + threadsperblock - arraysz%threadsperblock
    else:
        arrayszplus = arraysz

    # nonzero_ar is set to 1 for every element for which weight_ar is >0
    nonzero_ar = np.zeros((arrayszplus,), dtype=np.uint32)

    # scan_ar is going to hold the result of scanning the input
    scan_ar = np.zeros((arrayszplus,), dtype=np.uint32)

    # Explicitly copy working data to device (two lots of arraysz data on GPU memory)
    print ("Allocating " + str(4*arrayszplus) + " bytes on the GPU memory (d_nonzero_ar)")
    d_nonzero_ar = cuda.to_device (nonzero_ar)
    print ("Allocating " + str(4*arrayszplus) + " bytes on the GPU memory (d_scan_ar)")
    d_scan_ar = cuda.to_device (scan_ar)

    # Make up a list of carry vectors and allocate device memory
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

        print ("Allocating " + str(4*carrysz) + " bytes on the GPU memory (carrylist)")
        carrylist.append (np.zeros((carrysz,), dtype=np.float32))
        d_carrylist.append (cuda.to_device(carrylist[-1]))
        asz = math.ceil (asz / threadsperblock)
        scansz = asz
        if scansz%threadsperblock:
            scansz = scansz + threadsperblock - scansz%threadsperblock
        print ("Allocating " + str(4*scansz) + " bytes on the GPU memory (scanlist)")
        scanlist.append (np.zeros((scansz,), dtype=np.float32))
        d_scanlist.append (cuda.to_device(scanlist[-1]))

    # Add a last carrylist, as this will be required as a dummy carry list for the last call to one_block_scan()
    carrylist.append (np.zeros((1,), dtype=np.int32))
    d_carrylist.append (cuda.to_device(carrylist[-1]))

    #
    # Compute partial scans of the top-level weight_ar and the lower level
    # partial sums
    #
    # The first input is the weight array, compute block-wise prefix-scan sums:
    detect_nonzero[blockspergrid, threadsperblock] (d_weight_ar, d_nonzero_ar, arrayszplus)
    one_block_scan[blockspergrid, threadsperblock] (d_scan_ar, d_nonzero_ar, d_carrylist[0], threadsperblock, arrayszplus)

    asz = math.ceil (arrayszplus / threadsperblock)
    j = 0
    while asz > threadsperblock:
        scanblocks = math.ceil (asz / threadsperblock)
        scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
        one_block_scan[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))
        asz = scanblocks
        j = j+1
    # Plus one more iteration:
    scanblocks = math.ceil (asz / threadsperblock)
    scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
    one_block_scan[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))

    #
    # Construct the scans back up the tree by summing the "carry" into the "scans"
    #
    ns = len(scanlist)
    j = ns
    while j > 0:
        sumblocks = math.ceil(len(scanlist[j-1])/threadsperblock)
        sum_scans[sumblocks, threadsperblock](d_carrylist[j-1], d_scanlist[j-1], len(scanlist[j-1]), d_carrylist[j])
        # Now d_carrylist[j-1] has had its carrys added from the lower level
        j = j-1

    # The final sum_scans() call. Do I really need d_scanf_ar here? Couldn't I sum within d_scan_ar destructively at this point?
    #sum_scans[blockspergrid, threadsperblock](d_scanf_ar, d_scan_ar, arrayszplus, d_carrylist[0])
    sum_scans_destructively[blockspergrid, threadsperblock](d_scan_ar, arrayszplus, d_carrylist[0])


    # Finally, in parallel, populate d_result_idx and d_result_val.
    extract_nonzero[blockspergrid, threadsperblock] (d_weight_ar, arraysz, d_scan_ar, d_result_idx, d_result_val)

    return # END reduce_nonzero_gpu()
###############################################################################


#
# Example calling of reduce_nonzero_gpu
#

#rowlen =  26287 # 26287 rowlen uses up 8335607808 bytes of GPU
                # RAM. 1070 has 8502247424 bytes. The difference is
                # 166639616 bytes (166 MB) which must be required by
                # the code, and other variables.

rowlen = 162*162 # The max possible number that can be squared for rowlen.

threadsperblock = 128 # 128 is 1 Multiprocessor.

arraysz = rowlen*rowlen
# To pad the arrays out to exact number of blocks-worth of threads
if arraysz%threadsperblock:
    arrayszplus = arraysz + threadsperblock - arraysz%threadsperblock
else:
    arrayszplus = arraysz
print ("arraysz in exact number of blocks (arrayszplus) is " + str(arrayszplus))

weight_ar = np.zeros((arrayszplus,), dtype=np.float32)

# Now some non-zero, non-unary weights
weight_ar[0] = 1.1
weight_ar[1] = 1.3
weight_ar[2] = 1.2
weight_ar[3] = 0.2
weight_ar[4] = 0.3

if rowlen > 3:
    weight_ar[12] = 1.7
if rowlen > 4:
    weight_ar[22] = 1.9
if rowlen > 5:
    weight_ar[33] = 2.3
if rowlen > 17:
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

if rowlen > 149:
    weight_ar[22486] = 2.54

# Copy to GPU memory: (one lot of arraysz plus floats)
print ("Allocating " + str(4*arrayszplus) + " bytes on the GPU memory (d_weight_ar)")
d_weight_ar = cuda.to_device (weight_ar)

# Set up result arrays
result_sz = np.uint32(50)
result_idx = np.zeros((result_sz,), dtype=np.uint32)
d_result_idx = cuda.to_device (result_idx)
result_val = np.zeros((result_sz,), dtype=np.float32)
d_result_val = cuda.to_device (result_val)

# Call the function:
reduce_nonzero_gpu (d_weight_ar, arraysz, arrayszplus, threadsperblock, d_result_idx, d_result_val, result_sz)

# Retrieve results from memory
r_weight_ar = d_weight_ar.copy_to_host()
r_result_idx = d_result_idx.copy_to_host()
r_result_val = d_result_val.copy_to_host()

# Print result
j = 0
while j < result_sz:
    print ("result_idx[" + str(j) + "] = " + str(r_result_idx[j]) + " with value=" + str(r_result_val[j]))
    j = j+1
