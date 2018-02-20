#ifndef _ARB_RESAMPLER_H_
#define _ARB_RESAMPLER_H_

int set_num_samples(int *h_num_samples);
int set_last_sample(int *h_last_sample);

/*****************************************************************************/ /*!
 * @brief Sets all constant variables that are local on the Device
 * @param in double *h_delta Pointer to the accurate floating point
 *                           decimation rate
 * @param in double *h_accum The accumulated filtersteps so far
 * @param in double *h_flt_rate The floating point difference between h_delta
 *                              and integer decimation rate
 * @param in int    *h_num_filters The Number of filters in the filterbank
 * @param in int    *start_filter The first filter used inside the filterbank
 * @param in int    *h_num_taps The number of taps of each filter in the
 *                              filterbank
 * @param in float  *h_taps Pointer to the tap vector in Constant Memory
 * @param in float  *h_diff_taps Pointer to the differential tap vector in
 *                               Constant Memory
 ***************************************************************************/
int set_resampler_constants(double *h_delta, double *h_accum,
                            double *h_flt_rate, int *h_num_filters,
                            int *start_filter, int *h_num_taps,
                            int *h_channel_buffersize, float *h_taps,
                            float *h_diff_taps);

/*****************************************************************************/ /*!
 * @brief Updates the Startfilter on the Device
 * @param in int *h_start_filter Pointer to new starting filter
 ***************************************************************************/
int update_start_filter(int *h_start_filter);

/*****************************************************************************/ /*!
 * @brief Does the setup and firing of the Device Kernel
 * @param in float2 *inbuffer Pointer to input buffer in Global Device Memory
 * @param in float2 *outbuffer Pointer to output buffer in Global Device Memory
 * @param in float2 *history Pointer to the filter tail from prevoius runs
 *                           in Device Global Memory
 * @param in int    grid_dim Dimension of the Device Kernel Grid
 * @param in dim3   block_dim Dimension of Device Kernel Block
 * @param in size_t shared_mem_size Size of the Local Shared Device Memory
 ***************************************************************************/
void arb_resampler(float2 *inbuffer, float2 *outbuffer, float2 *history,
                   int grid_dim, dim3 block_dim, size_t shared_mem_size);

/*****************************************************************************/ /*!
 * @brief The Arbitrary Resampler CUDA Kernel
 * @param in float2 *inbuffer Pointer to input buffer in Global Device Memory
 * @param in float2 *outbuffer Pointer to output buffer in Global Device Memory
 * @param in float2 *history Pointer to the filter tail from prevoius runs
 *                           in Device Global Memory
 ***************************************************************************/
__global__ void arb_resampler_execute(float2 *in, float2 *out, float2 *history);

#endif /* ifndef _ARB_RESAMPLER_H_ */
