# Compute a widening Gaussian connection (GPU implementation)
#
# Parameters have the same meaning as in the CPU code except:
#
# max_n_weights is the size of the array to allocate to contain the
# generated weights. In principle this could be nfs^4, but for a 150 side
# neural field side, that would result in a requirement for 8 GB of device
# RAM. Instead, specify max_n_weights and hope you chose enough! 20,000,000
# is reasonable, but it will depend on the parameters of your projection.
def connectionFunc(srclocs,dstlocs,sigma_m,E2,sigma_0,fovshift,nfs,W_cut,osd0p,osd1r,max_n_weights):

    from numba import cuda, float32, int32
    import math
    import numpy as np
    from operator import gt
    import time # for code profiling

    # Ensure these are fixed to being ints
    _osd0p = int(osd0p)
    _osd1r = int(osd1r)

    # Shifts index to avoid bank conflicts (on GTX1070)
    @cuda.jit(device=True)
    def shifted_idx (idx):
        # 16 and 768 works for GTX1070/Compute capability 6.1; makes 48 KB of
        # shared memory. GTX 1080 May have 96 KB (double) but it's still CC 6.1
        num_banks = 16
        bank_width_int32 = 768 # max_idx = (bank_width_int32 * num_banks)
        idx_idiv_num_banks = idx // num_banks
        idx_mod_num_banks = idx % num_banks
        offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks))
        return offs_idx

    @cuda.jit(device=True)
    def shifted_idx3 (thread_idx):
        # How many int32 memory locations being used in each thread:
        memorywidth = 3 # 12//4
        bank_width_int32 = 192 # 768//4 (the width of a bank in int32s)
        bankwidth = 3072 # bank_width_int32 * num_banks (16)
        offs_idx = (bank_width_int32 * thread_idx)
        idx_idiv_bw = offs_idx // bankwidth
        idx_mod_bw = offs_idx % bankwidth
        offs_idx = idx_mod_bw + idx_idiv_bw * memorywidth
        return offs_idx

    ### GPU SCAN CODE GOES HERE
    ###############################################################################
    # Like prefixscan_gpu, but returning the non-zero values from weight_ar in
    # d_out
    #
    # d_weight_ar - A memory area on the GPU memory containing a sparse matrix
    # of results (floating point, probably)
    #
    # arraysz - The length of the sparse matrix contained in d_weight_ar (this
    # will be the same as nfs_sq * nfs_sq)
    #
    # arrayszplus - The size of the memory area d_weight_ar. This should be an
    # integer multiple of threadsperblock
    #
    # threadsperblock - how many threads to launch per threadblock on the GPU.
    #
    # d_out - Array for results in connectionFunc output format: 4 columns of
    # data. Populated by extract_nonzero()
    #
    # _nfs_sq square of neural field size (nfs is the side length of the square
    # neural field).
    #
    def reduce_nonzero_gpu (d_weight_ar, arraysz, arrayszplus, threadsperblock, _d_out, _nfs_sq):
        import math
        from operator import gt

        # Detect non-zero values in weight_ar_. Update each element of the
        # identically shaped nonzero_ar_ to hold 1 for non-zero values in
        # weight_ar_ and 0 for zero values in weight_ar_
        @cuda.jit("void(float32[:], int32[:], int32)")
        def detect_nonzero (weight_ar_, nonzero_ar_, arraysz):
            thid = cuda.threadIdx.x + (cuda.blockIdx.x*cuda.blockDim.x)
            if thid < arraysz:
                nonzero_ar_[thid] = 1 if weight_ar_[thid] > 0.0 else 0

        #
        # parallel prefix scan for stream compaction (See sect. 39.3
        # https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)
        #
        # This sums up the values in input_ar_, placing the running total in
        # scan_ar_. The final carry value is placed in carry_
        #
        # This kernel carries out the prefix scan for a single threadblock.
        #
        # scan_ar_ - The result of the prefix scan. Array of uint32s (could be
        # float32s)
        #
        # input_ar_ - The input for the algorithm. Array of uint32s
        #
        # carry_ - The carry array - carry_[cuda.blockIdx.x] is updated with
        # the final value in scan_ar_
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
                # ai and bi are well separated across the shared memory. bi =
                # ai+1 could also work?
                bi = ai + d

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

        # Last job is to add on the carry to each part of scan_ar WHILE AT THE
        # SAME TIME SUMMING WITHIN A BLOCK
        @cuda.jit
        def sum_scans(new_carry_ar_, scan_ar_, scan_ar_sz, carry_ar_):
            thid = cuda.threadIdx.x
            tb_offset = cuda.blockIdx.x*cuda.blockDim.x # threadblock offset
            arr_addr = thid+tb_offset

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

        # Extract non zero weights from d_weight_ar and place them into the array d_out.
        @cuda.jit
        def extract_nonzero (d_weight_ar, weight_sz, d_scan_ar, __d_out, __nfs_sq):
            thid = cuda.threadIdx.x + (cuda.blockIdx.x*cuda.blockDim.x)
            if thid < weight_sz and d_weight_ar[thid] > 0.0:
                # Populate d_out in the correct format:
                src_idx = thid%__nfs_sq
                dst_idx = thid//__nfs_sq
                jj = d_scan_ar[thid]
                __d_out[jj][0] = src_idx
                __d_out[jj][1] = dst_idx
                __d_out[jj][2] = 0.0 # delay - unused
                __d_out[jj][3] = d_weight_ar[thid]

        # END of kernel definitions

        # reduce_nonzero_gpu code proper starts:
        print ('--- reduce_nonzero_gpu ---')
        print ('threadsperblock: {0}'.format(threadsperblock))
        print ('arrayszplus: {0}'.format(arrayszplus))
        # Set by code above - size of the weight array. Quite big
        print ('arraysz: {0}'.format(arraysz))

        #
        # Build input data for the test
        #

        blockspergrid = math.ceil(arraysz/threadsperblock)
        print ('blockspergrid: {0}'.format(blockspergrid))

        # nonzero_ar is set to 1 for every element for which weight_ar is >0
        print ('allocate arrayszplus={0} uint32s in nonzero_ar'.format(arrayszplus))
        nonzero_ar = np.zeros((arrayszplus,), dtype=np.uint32)

        # scan_ar is going to hold the result of scanning the input
        scan_ar = np.zeros((arrayszplus,), dtype=np.uint32)

        # Explicitly copy working data to device (two lots of arraysz data on GPU memory)
        print ("Allocating "+str(4*arrayszplus/1048576)+" MBytes on the GPU (d_nonzero_ar)")
        d_nonzero_ar = cuda.to_device (nonzero_ar)
        print ("Allocating "+str(4*arrayszplus/1048576)+" MBytes on the GPU (d_scan_ar)")
        d_scan_ar = cuda.to_device (scan_ar)

        # Make up a list of carry vectors and allocate device memory
        carrylist = []
        d_carrylist = []
        # And containers for the scan
        scanlist = []
        d_scanlist = []
        asz = arraysz
        print ('asz: {0} threadsperblock: {1}'.format(asz, threadsperblock))
        while asz > threadsperblock:

            carrysz = math.ceil (asz / threadsperblock)
            # Ensure carrysz is a multiple of threadsperblock:
            if carrysz%threadsperblock:
                carrysz = carrysz + threadsperblock - carrysz%threadsperblock

            print ("Allocating "+str(4*carrysz/1024)+" KBytes on the GPU memory (carrylist)")
            carrylist.append (np.zeros((carrysz,), dtype=np.float32))
            d_carrylist.append (cuda.to_device(carrylist[-1]))
            asz = math.ceil (asz / threadsperblock)
            scansz = asz
            if scansz%threadsperblock:
                scansz = scansz + threadsperblock - scansz%threadsperblock
            print ("Allocating "+str(4*scansz/1024)+" KBytes on the GPU memory (scanlist)")
            scanlist.append (np.zeros((scansz,), dtype=np.float32))
            d_scanlist.append (cuda.to_device(scanlist[-1]))

        # Add a last carrylist, as this will be required as a dummy carry list
        # for the last call to one_block_scan()
        carrylist.append (np.zeros((1,), dtype=np.int32))
        d_carrylist.append (cuda.to_device(carrylist[-1]))

        # Compute partial scans of the top-level weight_ar and the lower level partial sums
        #
        # The first input is the weight array, compute block-wise prefix-scan sums:
        detect_nonzero[blockspergrid, threadsperblock] (d_weight_ar, d_nonzero_ar, arraysz)
        print ('First one_block_scan...')
        one_block_scan[blockspergrid, threadsperblock] (d_scan_ar, d_nonzero_ar, d_carrylist[0], threadsperblock, arrayszplus)

        asz = math.ceil (arrayszplus / threadsperblock)
        j = 0
        print ('asz: {0} threadsperblock: {1}'.format(asz, threadsperblock))
        while asz > threadsperblock:
            scanblocks = math.ceil (asz / threadsperblock)
            scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
            print ('while loop one_block_scan...')
            one_block_scan[scanblocks, threadsperblock](d_scanlist[j], d_carrylist[j], d_carrylist[j+1], threadsperblock, len(carrylist[j]))
            asz = scanblocks
            j = j+1
        # Plus one more iteration:
        scanblocks = math.ceil (asz / threadsperblock)
        scanblocks = scanblocks + threadsperblock - scanblocks%threadsperblock
        print ('last one_block_scan. scanblock: {0} threadsperblock: {1}, j is {2}'.format(scanblocks, threadsperblock, j))
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

        # The final sum_scans() call. I sum within d_scan_ar destructively at this point.
        sum_scans_destructively[blockspergrid, threadsperblock](d_scan_ar, arrayszplus, d_carrylist[0])

        # Get the total number of weights from the final carrylist:
        n_weights = 0
        local_carrylist = d_carrylist[ns].copy_to_host()
        last_cl_len = len(local_carrylist)
        if last_cl_len == 1:
            n_weights = local_carrylist[0]
        else:
            print ("ERROR. Length of last carry list should be 1")

        # Finally, in parallel, populate d_out.
        extract_nonzero[blockspergrid, threadsperblock] (d_weight_ar, arraysz, d_scan_ar, _d_out, _nfs_sq)

        return n_weights # END reduce_nonzero_gpu()
    ###############################################################################
    ### GPU SCAN CODE TO HERE

    ### CONNECTION FUNCTION-COMPUTING CODE HERE
    #
    @cuda.jit("void(float32,int32,  int32[:,:],int32[:,:],float32[:],float32, float32,float32, float32,  float32,float32, int32,     int32)")
    def dowork (M_f_start,  nfs_sq, d_src_ar,  d_dst_ar,  weight_ar, sigma_m, E2,     sigma_0, fovshift, nfs,    W_cut,   osd0p, osd1r):
        # Work out i_src and i_dst based on the 2-D thread index
        i_src, i_dst = cuda.grid(2)

        if i_src < nfs_sq and i_dst < nfs_sq:

            # Temporary shared memory for weights
            tmp_w = cuda.shared.array(12288, dtype=float32)
            myidx = (cuda.threadIdx.y*cuda.blockDim.x + cuda.threadIdx.x)
            offsidx = shifted_idx3 (myidx)
            tmp_w[offsidx] = float32(0.0)
            tmp_w[offsidx+1] = float32(0.0)
            tmp_w[offsidx+2] = float32(0.0)
            cuda.syncthreads()

            # Compute the location of d_src_ar, this defines what sigma will
            # be. As r (as opp. to phi) increases, the sigma should increase.
            M_f = float32(nfs)/(E2*math.log(((1+d_src_ar[i_src,1])/(2*E2))+1))

            # Set some of M_f to 1 to ensure the fan-out starts at around the
            # edge of the foveal region.
            if (1+d_src_ar[i_src,1]) < fovshift:
                M_f = M_f_start

            # Compute modified sigma and 3 times this value. _sigma is a
            # function of r, aka d_src_ar[1]. M_f is the function of r.
            _sigma = (sigma_m/M_f) - (sigma_m/M_f_start) + sigma_0
            three_sigma = float32(3.0) * _sigma

            # in-xy-plane distance (ignore d_src_ar[2]/dstdoc[2])
            xd = (d_src_ar[i_src,0] - d_dst_ar[i_dst,0] + osd0p)
            yd = (d_src_ar[i_src,1] - d_dst_ar[i_dst,1] + osd1r)
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
                tpb = cuda.blockDim.x * cuda.blockDim.y

                # Write data from tmp_w to res_ar, but only in ONE thread from
                # the threadblock. Should avoid racing.
                for idx in range(0,tpb): # 512 was hard coded here; changed it for tpb
                    offsidx2 = shifted_idx3 (idx)
                    theweight = tmp_w[offsidx2] # weight should be the first one, so no +1/+2
                    # Add to weight_ar
                    weight_idx = int32(tmp_w[offsidx2+2])*nfs_sq + int32(tmp_w[offsidx2+1])
                    weight_ar[weight_idx] = theweight

        return # end dowork()

    # Initialise a device array with a value. Use with a 1D grid of 1D threadblocks
    @cuda.jit("void(uint32[:],uint32,uint32)")
    def init_array (d_array, d_array_sz, value):
        thid = cuda.threadIdx.x + (cuda.blockIdx.x*cuda.blockDim.x)
        if thid < d_array_sz:
            d_array[thid] = value

    #
    # This is the start of the sequential connectionFunc code - equivalent to
    # the main() function in a C program.
    #

    # Compute some values once only:
    M_f_start=nfs/(E2*math.log((fovshift/(2*E2))+1))
    print('M_f_start: {0:f}'.format(M_f_start))
    nfs_sq = int(nfs*nfs)

    # Copy srclocs and dstlocs to device memory before starting.
    print('Create Numpy arrays...')
    src_ar = np.array(srclocs, dtype=np.int32)
    dst_ar = np.array(dstlocs, dtype=np.int32)
    print('Copy to device with cuda.to_device...')
    d_src_ar = cuda.to_device (src_ar)
    d_dst_ar = cuda.to_device (dst_ar)

    # Create a device array to have the weights computed into
    weight_sz = int(nfs_sq*nfs_sq) # Take care to cast all numbers to int
    d_weight_ar = cuda.device_array ((weight_sz,), dtype=np.float32)

    # Compute weights with dowork(), placing results into d_weight_ar
    threadsperblock = (int(16),int(32)) # 16 warps to a block; 512 threads
    print ('Compute weights: threadsperblock: {0}'.format(threadsperblock))
    blockspergrid = (int(1+(nfs_sq // threadsperblock[0])), int(1+(nfs_sq // threadsperblock[1])))
    print ('Compute weights: blockspergrid: {0}'.format(blockspergrid))
    time_start = int(round(time.time() * 1000))
    dowork[blockspergrid, threadsperblock](M_f_start, nfs_sq, d_src_ar,   d_dst_ar,   d_weight_ar, sigma_m, E2,     sigma_0, fovshift, nfs,     W_cut,  _osd0p, _osd1r)
    time_donework = int(round(time.time() * 1000))
    print ("computed weights after {0} ms".format(time_donework - time_start));

    # Extract Non-zero weights. For the reduce operation, I adopt a 1-D grid of threadblocks
    threadsperblock = int(128)
    blockspergrid = int(math.ceil(weight_sz/threadsperblock))
    if weight_sz%threadsperblock:
        weight_sz_plus = int(weight_sz + threadsperblock - weight_sz%threadsperblock)
    else:
        weight_sz_plus = int(weight_sz)

    # Allocate device memory for the reduce_nonzero_gpu result. In principle
    # this could be as large as nfs^4, but that can call for too much device
    # memory. A max_n_weights parameter of connectionFunc() is passed in to set
    # out_sz.
    out_sz = int(max_n_weights)

    # Allocate space for the output of reduce_nonzero_gpu() in device memory
    print ("Allocating " + str(4*4*out_sz/1048576) + " MBytes on the GPU memory (d_out)")
    d_out = cuda.device_array((out_sz,4), dtype=np.float32)

    dummy = 0
    print ("Reduce down to just the non-zero weights")
    n_weights = reduce_nonzero_gpu(d_weight_ar, weight_sz, weight_sz_plus, threadsperblock, d_out, nfs_sq)
    time_reduced = int(round(time.time() * 1000))
    print ("Completed reduce down after a further {0} ms. n_weights={1}".format(time_reduced-time_donework, n_weights))

    # Copy the device memory back with the result.
    out = d_out.copy_to_host()

    time_arraycreated = int(round(time.time() * 1000))
    print ("Got final result after a further {0} ms".format(time_arraycreated-time_reduced))

    print ("Total number of non-zero weights: {0}. out_sz: {1}.".format (n_weights, out_sz))
    if n_weights > max_n_weights:
        print ("---------------\nWARNING WARNING:\n---------------\n")
        print ("User chose {0} as max_n_weights, but {1} weights were generated.\n".format(max_n_weights, n_weights))
        print ("Memory corruption may have occurred!!\n---------------")

    # Truncate out at length n_weights. Note we're returning a Numpy array cast to a list.
    return out[0:n_weights,:].tolist() # end connectionFunc
#######################################################################
