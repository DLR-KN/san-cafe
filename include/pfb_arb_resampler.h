#ifndef __PFB_ARB_RESAMPLER_H__
#define __PFB_ARB_RESAMPLER_H__

#include <complex>
#include <memory>
#include <vector>

#include "archs.h"
#include "cafe_constants.h"
#include "cafe_cuda_helper.h"

namespace filter {

class pfb_arb_resampler {
  // Public memeber variables, public member functions are further down the file
 public:
  struct cuda_config {
    int rs_block_dimx;
    int rs_block_dimy;
    dim3 rs_grid_dim;
  };

 private:
  // Arbitrary Resampler variables
  int _num_filters;
  int _int_rate;
  double _delta;
  double _resample_rate;
  int _delta_samples;
  int _last_filter;
  double _accum;
  double _flt_rate;
  int _num_samples_kernel_out;
  int _num_samples_kernel_in;
  int _shared_mem_size;
  int _num_taps_per_filter;
  std::vector<float> _prot_taps;
  std::vector<float> _poly_taps;
  std::vector<float> _diff_taps;

  // CUDA Grid Variables
  enum { warp_size = 32 };
  int _num_channels;
  int _num_threadblocks;
  int _num_threads_in_block;
  int _cuda_shared_mem_size;
  dim3 _filter_block_config;
  cuda_config _grid_config;

  // CUDA GPU_BUFFERS
  int _cuda_buffer_len;
  cafe::cuda_unique_ptr _history;
  pfb_cuda_config _cuda_config;

 public:
  typedef std::shared_ptr<pfb_arb_resampler> pfb_arb_resampler_sptr;

  pfb_arb_resampler(const std::vector<float> &taps, int num_filters,
                    double resample_rate, int num_channels);
  ~pfb_arb_resampler();

  static pfb_arb_resampler_sptr pfb_arb_resampler_factory(
      const std::vector<float> &taps, int num_filters, double resample_rate,
      int num_channels);
  void set_taps(const std::vector<float> &prot_taps,
                std::vector<float> &poly_taps, int num_filters);
  void set_diff_taps(const std::vector<float> &taps,
                     std::vector<float> &difftaps);
  void set_filters();
  void set_grid();
  void set_grid(int block_dimx, int block_dimy, dim3 grid_dim);
  void set_indexing(double rate, int num_filters);
  int gen_index_plan(int len_inbuffer);
  void set_constant_symbols();
  void set_history();
  void set_num_samples_gpu(int num_samples);
  int set_filter_block_config(unsigned int block_size);

  int filter(int num_insamples, float2 *input_buffer, float2 *output_buffer);
};

} /* namespace filter */

#endif /* __PFB_ARB_RESAMPLER_H__ */
