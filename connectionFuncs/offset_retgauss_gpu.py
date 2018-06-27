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

    ### GPU SCAN CODE GOES HERE
    ###############################################################################
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
    ### GPU SCAN CODE TO HERE

    # Shifts index to avoid bank conflicts (on GTX1070)
    @cuda.jit(device=True)
    def shifted_idx_dup (idx):
        num_banks = 16 # 16 and 768 works for GTX1070/Compute capability 6.1
        bank_width_int32 = 768
        # max_idx = (bank_width_int32 * num_banks)
        idx_idiv_num_banks = idx // num_banks
        idx_mod_num_banks = idx % num_banks
        offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks))
        return offs_idx

    ### CONNECTION FUNCTION-COMPUTING CODE HERE
    @cuda.jit#("void(float32, int32, float32[:,:], float32[:,:], float32[:], float32,float32,float32,float32,float32,float32,int32,int32)")
    def dowork (M_f_start, nfs_sq, src_ar, dst_ar, weight_ar, sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r):
        # Work out i_src and i_dst based on the 2-D thread index
        i_src, i_dst = cuda.grid(2)
        if i_src < nfs_sq and i_dst < nfs_sq:

            # Temporary shared memory for weights
            tmp_w = cuda.shared.array(12288, dtype=float32) # Note - allocating ALL shared memory here.
            myidx = (cuda.threadIdx.y*cuda.blockDim.x + cuda.threadIdx.x)
            offsidx = shifted_idx_dup (myidx)
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
                    offsidx = shifted_idx_dup (idx)
                    theweight = tmp_w[offsidx]
                    #if gt(theweight, W_cut): # Testing makes no difference to speed
                    # Add to weight_ar
                    weight_idx = int32(tmp_w[offsidx+1])*nfs_sq + int32(tmp_w[offsidx+2])
                    weight_ar[weight_idx] = theweight

        return # end dowork()

    # Compute once only
    M_f_start=nfs/(E2*math.log((fovshift/(2*E2))+1))
    nfs_sq = nfs*nfs

    # Copy srclocs and dstlocs to device memory before starting
    src_ar = np.array(srclocs, dtype=np.int32)
    dst_ar = np.array(dstlocs, dtype=np.int32)

    # Device array to have the weights computed into
    weight_sz = nfs_sq*nfs_sq
    d_weight_ar = cuda.device_array((weight_sz,), dtype=np.float32)

    # COMPUTE WEIGHTS. Call function to compute weights in d_weight_ar
    threadsperblock = (16,32) # 16 warps to a block; 512 threads
    blockspergrid = (1+(nfs_sq // threadsperblock[0]), 1+(nfs_sq // threadsperblock[1]))
    print ("dowork...");
    dowork[blockspergrid, threadsperblock](M_f_start, nfs_sq, src_ar, dst_ar, d_weight_ar, sigma_m,E2,sigma_0,fovshift,nfs,W_cut,offsetd0p,offsetd1r)
    print ("done work");

    # EXTRACT NONZERO WEIGHTS. For the reduce operation, I adopt a 1-D grid of threadblocks
    threadsperblock = 512
    blockspergrid = math.ceil(weight_sz/threadsperblock)
    if weight_sz%threadsperblock:
        weight_sz_plus = weight_sz + threadsperblock - weight_sz%threadsperblock
    else:
        weight_sz_plus = weight_sz

    # Allocate device memory for the reduce_nonzero_gpu result
    out_sz = 2400 # 2400 enough for rowlen 50
    d_result_idx = cuda.device_array((out_sz,), dtype=np.uint32)
    d_result_val = cuda.device_array((out_sz,), dtype=np.float32)

    # Reduce it down
    dummy = 0
    reduce_nonzero_gpu(d_weight_ar, weight_sz, weight_sz_plus, threadsperblock, d_result_idx, d_result_val, dummy)

    r_result_idx = d_result_idx.copy_to_host()
    r_result_val = d_result_val.copy_to_host()

    # Create the array for the final output
    out = np.zeros((out_sz,4), dtype=np.float32)

    # Populate it from r_result_idx/val:
    j = 0
    k = 0
    # This is slooow
    while j < out_sz:
        src_idx = r_result_idx[j]%nfs
        dst_idx = r_result_idx[j]//nfs
        out[j][0] = src_idx
        out[j][1] = dst_idx
        out[j][2] = 0.0 # delay - unused
        out[j][3] = r_result_val[j]
        if (src_idx > 0 or dst_idx > 0):
            k = k+1
            print ("src_idx: " + str(src_idx) + " dst_idx: " + str(dst_idx) + " weight: " + str(out[j][3]))
        j = j+1

    print("Result array shape " + str(out.shape) + " k was " + str(k))

    return out # end connectionFunc
#######################################################################


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
show_graphs = 1
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
