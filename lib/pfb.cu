#include <filterbank_execute.h>
#include <pfb.h>
#include <stream_to_streams.h>
#include <iostream>

namespace cuda
{
void pfb_execute(float2 *in_stream, float2 *in_streams, float2 *fft_in,
                 float2 *fft_out, pfb_cuda_config *cuda_config,
                 size_t shared_mem_size)
{
  dim3 shuffle_blockconfig(cuda_config->shuffle_blockdim_x,
                           cuda_config->shuffle_blockdim_y, 1);
  dim3 fb_blockconfig(cuda_config->fb_blockdim_x, cuda_config->fb_blockdim_y,
                      1);
  stream_to_streams<<<cuda_config->shuffle_griddim, shuffle_blockconfig>>>(
      in_stream, in_streams);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Shuffle kernel failed with message : "
              << cudaGetErrorString(err) << "\n";
  }

  filterbank_execute<<<cuda_config->fb_griddim, fb_blockconfig,
                       shared_mem_size>>>(in_streams, fft_in);
  cudaDeviceSynchronize();

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Filterbank kernel failed with message : "
              << cudaGetErrorString(err) << "\n";
  }
}

void shuffle_input(float2 *in_stream, float2 *in_streams,
                   pfb_cuda_config *cuda_config)
{
  streams_to_stream<<<cuda_config->shuffle_griddim,
                      cuda_config->shuffle_blockdim_x>>>(in_stream, in_streams);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Shuffle kernel failed with message : "
              << cudaGetErrorString(err) << "\n";
  }
}

void synth_filter_shuffle(float2 *in_stream, float2 *out_streams,
                          int grid_dimension, int block_dimension_x,
                          int block_dimension_y)
{
  dim3 block_config(block_dimension_x, block_dimension_y, 1);
  stream_to_streams<<<grid_dimension, block_config>>>(in_stream, out_streams);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Synthesis Filter Shuffle failed with message: "
              << cudaGetErrorString(err) << "\n";
  }
}

void anti_imaging_filter(float2 *in_samples, float2 *out_samples,
                         pfb_cuda_config *cuda_config, int shared_mem_size)
{
  filterbank_no_os_execute<<<cuda_config->fb_griddim,
                             cuda_config->fb_blockdim_x, shared_mem_size>>>(
      in_samples, out_samples);
  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    std::cout << "Filterbank kernel failed with message : "
              << cudaGetErrorString(err) << "\n";
  }
}

} /* end namespace cuda */
