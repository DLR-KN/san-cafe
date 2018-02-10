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

namespace filter
{
enum {
  default_fb_grid_dim_y = 8
};  //! Compile time constant filterbank grid Y dimension
enum {
  default_sh_grid_dim = 16
};  //! Compile time constant shuffle grid dimension

/*!
 *  @brief The Implementation of the Polyphase Filterbank Arbitrary Resampler
 *         This class provides means to setup a PFB Arbitrary Resampler Filter
 *         on the device and functions to execute it
******************************************************************************/
class pfb_channelizer_ccf
{
 private:
  pfb_cuda_config _cuda_config;  //! Cuda Grid and Block Config

  std::vector<float> _p_taps;  //! Prototype filter taps

  int _num_channels;         //! Number of channels to be extracted
  int _oversampling;         //! Oversampling factor of output
  int _num_taps_per_filter;  //! Number of taps per channel
  int _cuda_buffer_len;      //! Length of the output buffer

  // Grid variables
  enum { _warp_size = 32 };   //! Compile time constant Warp Size
  int _num_thread_blocks;     //! Number of blocks in grid
  int _threads_per_block;     //! Number of threads per block
  int _cuda_shared_mem_size;  //! Size of the cuda Shared memory
  int _inbuffer_size;         //! Size of the input buffers on the device
  int _samples_to_process;    //! Samples to process per filter call
  int _pfb_produce;           //! Samples produced per filter call

  // CUDA GPU Buffers
  cafe::cuda_unique_ptr _g_instream;     //! Input Stream on the GPU
  cafe::cuda_unique_ptr _g_instreams;    //! Streams on the GPU
  cafe::cuda_unique_ptr _fft_inbuffer;   //! fft input buffer on the GPU
  cafe::cuda_unique_ptr _fft_outbuffer;  //! fft output buffer from the gpu

  // CUFFT Plan
  cufftHandle _fft_plan;  //! Execution plan for the CUDA CuFFT

  //
  std::complex<float> *_fft_out;  //! Output of the fft on the host

 public:
  //! Typedef for the shared ptr of the pfb_channelizer class
  typedef std::shared_ptr<pfb_channelizer_ccf> pfb_channelizer_sptr;

  /*****************************************************************************/ /*!
   * @brief Constructor for the pfb_channelizer class
   * @param in std::vector<float> &taps Vector with prototype filter taps
   * @param in int channels     Number of channels to be processed
   * @param in int oversampling Oversampling rate of the PFB output
   ***************************************************************************/
  pfb_channelizer_ccf(const std::vector<float> &taps, int channels,
                      int oversampling);
  /*****************************************************************************/ /*!
   * @brief Destructor for the pfb_channelizer class
   ***************************************************************************/
  ~pfb_channelizer_ccf();

  /*****************************************************************************/ /*!
   * @brief Factory function for the pfb_channelizer class
   * @param in std::vector<float> &taps Vector with prototype filter taps
   * @param in int channels     Number of channels to be processed
   * @param in int oversampling Oversampling rate of the PFB output
   ***************************************************************************/
  static pfb_channelizer_sptr pfb_channelizer_factory(
      const std::vector<float> &taps, int channels, int oversampling);

  /*****************************************************************************/ /*!
   * @brief Sets up the taps to be used
   * @param in std::vector<float> &prot_taps Vector with prototype filter taps
   * @param in int channels Number of channels processed
   ***************************************************************************/
  void set_filters(const std::vector<float> &taps, int channels);

  /*****************************************************************************/ /*!
   * @brief Sets the streams for the shuffle kernel
   ***************************************************************************/
  void set_streams();

  /*****************************************************************************/ /*!
   * @brief Sets the CUDA grids for all kernels on the device
   ***************************************************************************/
  void set_grid();

  /*****************************************************************************/ /*!
   * @brief Sets some of the constant values used in the CUDA Kernels
   ***************************************************************************/
  void set_constant_symbols();

  /*****************************************************************************/ /*!
   * @brief Get the PFB taps
   ***************************************************************************/
  std::vector<float> taps() const;

  /*****************************************************************************/ /*!
   * @brief Get the number of channels
   ***************************************************************************/
  int get_channels() { return _num_channels; };

  /*****************************************************************************/ /*!
   * @brief Debug Print all taps
   ***************************************************************************/
  void print_taps();

  /*****************************************************************************/ /*!
   * @brief Executes the CUDA Channelizer kernel on the device
   * @param in std::complex<float> *input_buffer Pointer to input buffer on
   *                                              device Global Memory
   * @param in std::complex<float> *output_buffer Pointer to output buffer on
   *                                              device Global Memory
   * @param in int num_insamples Numbers of samples to be processed
   ***************************************************************************/
  int filter(std::complex<float> *input, std::complex<float> *output,
             unsigned int num_samples);
};

}  // Namespace filter
#endif /* __PFB_CHANNELIZER_H__  */
