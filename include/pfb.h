#ifndef _PFB_H_
#define _PFB_H_
#include <cafe_constants.h>
#include <cuda_runtime.h>

namespace cuda {

void pfb_execute(float2 *in_stream, float2 *in_streams, float2 *fft_in,
                 float2 *fft_out, pfb_cuda_config *cuda_config,
                 size_t shared_mem_size);

void shuffle_input(float2 *in_stream, float2 *in_streams,
                   pfb_cuda_config *cuda_config);

void synth_filter_shuffle(float2 *in_stream, float2 *out_streams,
                          int grid_dimension, int block_dimension_x,
                          int block_dimension_y);

void anti_imaging_filter(float2 *in_samples, float2 *out_samples,
                         pfb_cuda_config *cuda_config, int shared_mem_size);

}  // namespace cuda

#endif /* _PFB_H_ */
