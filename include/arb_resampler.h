#ifndef _ARB_RESAMPLER_H_
#define _ARB_RESAMPLER_H_

int set_num_samples(int *h_num_samples);
int set_last_sample(int *h_last_sample);

int set_resampler_constants(double *h_delta, double *h_accum,
                            double *h_flt_rate, int *h_num_filters,
                            int *start_filter, int *h_num_taps, float *h_taps,
                            float *h_diff_taps, int *h_num_samples);

int update_start_filter(int *h_start_filter);

void arb_resampler(float2 *inbuffer, float2 *outbuffer, float2 *history,
                   int grid_dim, dim3 block_dim, size_t shared_mem_size);

__global__ void arb_resampler_execute(float2 *in, float2 *out, float2 *history);

#endif /* ifndef _ARB_RESAMPLER_H_ */
