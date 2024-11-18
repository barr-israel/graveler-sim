#include <curand_kernel.h>
#define BLOCKSIZE 1024
#define WARPSIZE 32

__global__ void rng(int *max_block_arr, int seed) {
  static const int MASK = !((1 << 25) - 1);
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int warpIdx = threadIdx.x / WARPSIZE;
  unsigned int index_in_warp = threadIdx.x % WARPSIZE;
  __shared__ unsigned char max_warp_arr[WARPSIZE];
  curandState state;
  curand_init(seed + index, 0, 0, &state);
  long runs = 500000000UL / (blockDim.x * gridDim.x) +
              1; // +1 to make up for truncation
  int max_t = 0;
  for (int i = 0; i <= runs; i++) {
    int count1 = 0;
    count1 += __popc(curand(&state) & curand(&state)); // 32
    count1 += __popc(curand(&state) & curand(&state)); // 64
    count1 += __popc(curand(&state) & curand(&state)); // 96
    count1 += __popc(curand(&state) & curand(&state)); // 128
    count1 += __popc(curand(&state) & curand(&state)); // 160
    count1 += __popc(curand(&state) & curand(&state)); // 192
    count1 += __popc(curand(&state) & curand(&state)); // 224
    int count2 = 0;
    count2 += __popc(curand(&state) & curand(&state)); // 32
    count2 += __popc(curand(&state) & curand(&state)); // 64
    count2 += __popc(curand(&state) & curand(&state)); // 96
    count2 += __popc(curand(&state) & curand(&state)); // 128
    count2 += __popc(curand(&state) & curand(&state)); // 160
    count2 += __popc(curand(&state) & curand(&state)); // 192
    count2 += __popc(curand(&state) & curand(&state)); // 224
    unsigned int final_set = curand(&state);
    count1 += __popc(final_set & final_set << 7 & MASK);        // 231
    count2 += __popc(final_set << 14 & final_set << 21 & MASK); // 231
    max_t = max(max_t, max(count1, count2));
  }
  __syncwarp();
  // intra-warp reduction
  max_t = max(max_t, __shfl_down_sync(0xFFFFFFFF, max_t, 16));
  max_t = max(max_t, __shfl_down_sync(0xFFFFFFFF, max_t, 8));
  max_t = max(max_t, __shfl_down_sync(0xFFFFFFFF, max_t, 4));
  max_t = max(max_t, __shfl_down_sync(0xFFFFFFFF, max_t, 2));
  max_t = max(max_t, __shfl_down_sync(0xFFFFFFFF, max_t, 1));
  if (index_in_warp == 0) {
    max_warp_arr[warpIdx] = max_t;
  }
  __syncthreads();
  if (warpIdx == 0) { // reduce all other warps in the block to one value
    unsigned char max_block = max_warp_arr[index_in_warp];
    max_block = max(max_block, __shfl_down_sync(0xFFFFFFFF, max_block, 16));
    max_block = max(max_block, __shfl_down_sync(0xFFFFFFFF, max_block, 8));
    max_block = max(max_block, __shfl_down_sync(0xFFFFFFFF, max_block, 4));
    max_block = max(max_block, __shfl_down_sync(0xFFFFFFFF, max_block, 2));
    max_block = max(max_block, __shfl_down_sync(0xFFFFFFFF, max_block, 1));
    max_block_arr[blockIdx.x] = max_block;
  }
}
