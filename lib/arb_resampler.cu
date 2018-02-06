#define MAX_FILTER_SIZE 1024

#include <iostream>

__constant__ float delta;
__constant__ float accum;
__constant__ float flt_rate;
__constant__ int num_filters;
__constant__ int start_filter;
__constant__ int num_taps;
__constant__ int num_samples;
__constant__ int last_sample;
__constant__ float taps[MAX_FILTER_SIZE];
__constant__ float diff_taps[MAX_FILTER_SIZE];

int set_num_samples(int *h_num_samples)
{
  int ret = cudaMemcpyToSymbol(num_samples, h_num_samples, sizeof(float));
  return ret;
}

int set_last_sample(int *h_last_sample)
{
  int ret = cudaMemcpyToSymbol(last_sample, h_last_sample, sizeof(float));
  return ret;
}

int set_resampler_constants(double *h_delta, double *h_accum,
                            double *h_flt_rate, int *h_num_filters,
                            int *h_start_filter, int *h_num_taps, float *h_taps,
                            float *h_diff_taps, int *h_num_samples)
{
  int ret = 0;

  float delta_f = (float)*h_delta;
  float accum_f = (float)*h_accum;
  float flt_rate_f = (float)*h_flt_rate;

  ret = cudaMemcpyToSymbol(delta, &delta_f, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(accum, &accum_f, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(start_filter, h_start_filter, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(flt_rate, &flt_rate_f, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(num_filters, h_num_filters, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(num_taps, h_num_taps, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(num_samples, h_num_samples, sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(diff_taps, h_diff_taps,
                           ((*h_num_taps) * (*h_num_filters)) * sizeof(float));
  if (ret) return ret;

  ret = cudaMemcpyToSymbol(taps, h_taps,
                           ((*h_num_taps) * (*h_num_filters)) * sizeof(float));
  if (ret) return ret;

  return ret;
}

int update_start_filter(int *h_start_filter)
{
  int ret = 0;
  ret = cudaMemcpyToSymbol(start_filter, h_start_filter, sizeof(float));
  return ret;
}

__global__ void arb_resampler_execute(float2 *in, float2 *out, float2 *history)
{
  extern __shared__ float2 shared_mem[];
  float2 *delay_line = (float2 *)&shared_mem[0];
  float2 result = make_float2(0.0, 0.0);
  float2 diff_result = make_float2(0.0, 0.0);

  int operation = threadIdx.x + threadIdx.y * blockDim.x;

  int filter = ((int)(operation * delta));
  filter = (filter + start_filter) % num_filters;
  int sample = ((int)(operation * delta) + start_filter) / num_filters;
  int num_threads = blockDim.x * blockDim.y;
  float acc = operation * flt_rate + accum;
  acc = fmodf(acc, 1.0);

  // Copy shared "history" over to shared memory
  if (operation < num_taps - 1) {
    delay_line[operation] = history[blockIdx.x * (num_taps - 1) + operation];
  }
  __syncthreads();

  // Copy sample to shared memory
  int samples_left = num_samples - num_threads;
  if (operation < num_samples) {
    delay_line[operation + num_taps - 1] =
        in[blockIdx.x * num_samples + operation];
  }
  __syncthreads();

  // Copy all samples left
  if (operation < samples_left) {
    delay_line[operation + num_taps - 1 + num_threads] =
        in[blockIdx.x * num_samples + operation + num_threads];
  }
  __syncthreads();

  // Do the dotproduct
  for (int i = 0; i < num_taps; ++i) {
    result.x +=
        delay_line[num_taps - 1 + sample - i].x * taps[filter * num_taps + i];
    result.y +=
        delay_line[num_taps - 1 + sample - i].y * taps[filter * num_taps + i];
    diff_result.x +=
        delay_line[num_taps + sample - i].x * diff_taps[filter * num_taps + i];
    diff_result.y +=
        delay_line[num_taps + sample - i].y * diff_taps[filter * num_taps + i];
  }

  // Copy stuff back to global memory
  out[blockIdx.x * blockDim.x * blockDim.y + operation].x =
      result.x + diff_result.x * acc;
  out[blockIdx.x * blockDim.x * blockDim.y + operation].y =
      result.y + diff_result.y * acc;

  // Copy the history back to global memory
  if (operation < num_taps - 1) {
    history[blockIdx.x * (num_taps - 1) + operation] =
        delay_line[last_sample - num_taps + 1 + operation];
  }
}

void arb_resampler(float2 *inbuffer, float2 *outbuffer, float2 *history,
                   int grid_dim, dim3 block_dim, size_t shared_mem_size)
{
  cudaGetLastError();
  dim3 grid(grid_dim);
  arb_resampler_execute<<<grid, block_dim, shared_mem_size>>>(
      inbuffer, outbuffer, history);
  cudaDeviceSynchronize();

  // Check for errors while executing the kernel
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Resample Kernel failed with message : "
              << cudaGetErrorString(err) << "\n";
  }
}
