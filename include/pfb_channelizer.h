#ifndef __PFB_CHANNELIZER_H__
#define __PFB_CHANNELIZER_H__

#include <complex>
#include <memory>
#include <vector>

#include <arch_types.h>
#include <archs.h>

#include <cafe_constants.h>
#include <cafe_cuda_helper.h>
#include <cufft.h>

namespace filter {

enum { default_fb_grid_dim_y = 8 };
enum { default_sh_grid_dim = 16 };

// Class for FFT filter with complex in/output and real float taps
class pfb_channelizer_ccf {
 private:
  pfb_cuda_config _cuda_config;

  std::vector<float> _p_taps;

  int _num_channels;
  int _oversampling;
  int _num_taps_per_filter;
  int _cuda_buffer_len;

  // Grid variables
  static const int _warp_size = 32;
  int _num_thread_blocks;
  int _threads_per_block;
  int _cuda_shared_mem_size;
  int _inbuffer_size;
  int _samples_to_process;
  int _pfb_produce;

  // CUDA GPU Buffers
  cafe::cuda_unique_ptr _g_instream;     // Input Stream on the GPU
  cafe::cuda_unique_ptr _g_instreams;    // Streams on the GPU
  cafe::cuda_unique_ptr _fft_inbuffer;   // fft input buffer on the GPU
  cafe::cuda_unique_ptr _fft_outbuffer;  // fft output buffer from the gpu

  // CUFFT Plan
  cufftHandle _fft_plan;

  //
  std::complex<float> *_fft_out;

 public:
  typedef std::shared_ptr<pfb_channelizer_ccf> pfb_channelizer_sptr;

  pfb_channelizer_ccf(const std::vector<float> &taps, int channels,
                      int oversampling);
  ~pfb_channelizer_ccf();

  static pfb_channelizer_sptr pfb_channelizer_factory(
      const std::vector<float> &taps, int channels, int oversampling);

  void set_filters(const std::vector<float> &taps, int channels);
  void set_streams();
  void set_grid();

  /*!
   * \brief Sets some of the constant values used in the CUDA Kernels such as
   * \param samples_per_stream The number of samples per filter row
   * \param shift_width The number of samples the input streams are shifted
   *
   */
  void set_constant_symbols();

  std::vector<float> taps() const;
  int get_channels() { return _num_channels; };
  void print_taps();

  int filter(std::complex<float> *input, std::complex<float> *output,
             unsigned int num_samples);
};

}  // Namespace filter
#endif /* __PFB_CHANNELIZER_H__  */
