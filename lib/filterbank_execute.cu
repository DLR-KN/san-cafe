#include <filterbank_execute.h>

#define MAX_FILTER_SIZE 4096

__constant__ int shift_width;
__constant__ int offset;
__constant__ int num_taps;
__constant__ float taps[MAX_FILTER_SIZE];  // We can not allocate the memory
                                           // dynamically so we allocate the
                                           // maximum amount of bytes we are
                                           // willing to make

int copy_filter_taps(float *h_taps, int prot_filter_size)
{
  return cudaMemcpyToSymbol(taps, h_taps, sizeof(float) * prot_filter_size);
}

int set_filterbank_constants(int *h_offset, int *h_shift_width, float *h_taps,
                             int *taps_per_channel, int prot_filter_size)
{
  int ret = 0;
  ret = cudaMemcpyToSymbol(offset, h_offset, sizeof(int));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(shift_width, h_shift_width, sizeof(int));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(num_taps, taps_per_channel, sizeof(int));
  if (ret) return ret;

  ret = copy_filter_taps(h_taps, prot_filter_size);
  return ret;
}

__global__ void filterbank_execute(float2 *in, float2 *out)
{
  extern __shared__ float2 shared_mem[];

  float2 result = make_float2(0.0, 0.0);

  float2 *delay_line = (float2 *)&shared_mem[0];

  // The offset inside the input buffer
  int start_offset = blockIdx.y * blockDim.x;

  // The shift of the filter relative to the sample
  int filter_shift = blockIdx.x - (threadIdx.y + 1) * shift_width;

  // The actual filterbank number but not modulo 45, determines the starting
  // sample
  int filter_total = gridDim.x - 1 - filter_shift;

  // The filterbank used for this sample
  int filter = filter_total % gridDim.x;

  // An additional shift due to the input shifting snakelike through the
  // filterbank
  int sample = filter_shift >= 0;

  // Copy the data to shared memory. Because we have the need for speed!
  if (threadIdx.y == 0) {
    delay_line[threadIdx.x] =
        in[offset * blockIdx.x + start_offset + threadIdx.x];
  }

  if (threadIdx.x < num_taps) {
    delay_line[blockDim.x + threadIdx.x] =
        in[offset * blockIdx.x + start_offset + blockDim.x + threadIdx.x];
  }

  __syncthreads();

  // Do the dot product
  for (int i = 0; i < num_taps; ++i) {
    result.x += delay_line[num_taps + threadIdx.x - i - sample].x *
                taps[filter * num_taps + i];
    result.y += delay_line[num_taps + threadIdx.x - i - sample].y *
                taps[filter * num_taps + i];
  }

  // Calculate the offset for the despinning. This has to happen if the
  // oversampling is  > 1
  int fft_pos = gridDim.x - ((shift_width + blockIdx.x) % gridDim.x) - 1;
  // Copy the result back to global memory at the appropriate position
  out[fft_pos +
      gridDim.x * (threadIdx.y + threadIdx.x * blockDim.y +
                   blockIdx.y * blockDim.x * blockDim.y)] = result;
}

__global__ void filterbank_no_os_execute(float2 *in, float2 *out)
{
  extern __shared__ float2 shared_mem[];

  float2 result = make_float2(0.0, 0.0);

  float2 *delay_line = (float2 *)&shared_mem[0];

  // The offset inside the input buffer

  // The filterbank used for this sample
  int filter = blockIdx.x;

  // Copy the data to shared memory. Because we have the need for speed!
  delay_line[threadIdx.x] = in[offset * blockIdx.x + threadIdx.x];

  if (threadIdx.x < num_taps) {
    delay_line[blockDim.x + threadIdx.x] =
        in[offset * blockIdx.x + blockDim.x + threadIdx.x];
  }

  __syncthreads();

  // Do the dot product
  for (int i = 0; i < num_taps; ++i) {
    result.x += delay_line[num_taps + threadIdx.x - i - 1].x *
                taps[filter * num_taps + i];
    result.y += delay_line[num_taps + threadIdx.x - i - 1].y *
                taps[filter * num_taps + i];
  }

  // Copy the result back to global memory at the appropriate position
  // Not optimal but should be enough and it is the most simple solution
  out[gridDim.x * threadIdx.x + blockIdx.x] = result;
}
