from numba import cuda
print(cuda.gpus)

# GP104 (1070, 1080, compute capability 6.1). 32 banks of 768 bytes
# gives 24576 bytes per 32 cores. There are 128 cores per
# multiprocessor and 24576*4 is 96 KB.

# Shifts index to avoid bank conflicts (on GTX 1070 or GTX 1080)
#
# 0 maps to 0   # Note you can write to locations 0, 1, 2 (with 32 bit words)
# 1 maps to 192 # Write to 192, 193, 194 on the next bank, so in the next thread.
# 2 maps to 384
# 3 maps to 576
# 4 maps to 768
# 5 maps to 960
# 6 maps to 1152
# 7 maps to 1344
# 8 maps to 1536
# 9 maps to 1728
# 10 maps to 1920
# 11 maps to 2112
# 12 maps to 2304
# 13 maps to 2496
# 14 maps to 2688
# 15 maps to 2880
# 16 maps to 3072
# 17 maps to 3264
# 18 maps to 3456
# 19 maps to 3648
# 20 maps to 3840
# 21 maps to 4032
# 22 maps to 4224
# 23 maps to 4416
# 24 maps to 4608
# 25 maps to 4800
# 26 maps to 4992
# 27 maps to 5184
# 28 maps to 5376
# 29 maps to 5568
# 30 maps to 5760
# 31 maps to 5952
# 32 maps to 3     # Note you can write to locations 3, 4, 5 (with 32 bit words)
# 33 maps to 195
# 34 maps to 387
# 35 maps to 579
# etc
#
@cuda.jit(device=True)
def shifted_idx3 (thread_idx):
    # How many int32 memory locations being used in each thread:
    memorywidth = 3 # 12//4
    # The width of a bank in int32s:
    bank_width_int32 = 192 # 768//4
    #num_banks = 16
    bankwidth = 3072 # bank_width_int32 * num_banks

    offs_idx = (bank_width_int32 * thread_idx)
    idx_idiv_bw = offs_idx // bankwidth
    idx_mod_bw = offs_idx % bankwidth
    offs_idx = idx_mod_bw + idx_idiv_bw * memorywidth

    return offs_idx

for i in range(0,36):
    #for j in range(0,3):
    print ('{0} maps to {1}'.format(i,shifted_idx3(i)))
