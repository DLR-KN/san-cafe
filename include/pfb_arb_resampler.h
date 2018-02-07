#ifndef __PFB_ARB_RESAMPLER_H__
#define __PFB_ARB_RESAMPLER_H__

#include <complex>
#include <memory>
#include <vector>

#include "archs.h"
#include "cafe_constants.h"
#include "cafe_cuda_helper.h"

namespace filter
{
/*!
 *  @brief The Implementation of the Polyphase Filterbank Arbitrary Resampler
 *         This class provides means to setup a PFB Arbitrary Resampler Filter
 *         on the device and functions to execute it
******************************************************************************/
class pfb_arb_resampler
{
  // Public memeber variables, public member functions are further down the file
 public:
  struct cuda_config {  //! Struct to hold all the Resampler CUDA Grid/Block
                        //! Dimensions
    int rs_block_dimx;  //! Resampler Block X Dimension
    int rs_block_dimy;  //! Resampler Block Y Dimension
    dim3 rs_grid_dim;   //! Resampler Grid Dimension
  };

 private:
  // Arbitrary Resampler variables
  int _num_filters;       //! Number of filters and Interpolation Rate
  int _int_rate;          //! Integer Decimation Rate
  double _delta;          //! Difference between _resample_rate and _int_rate
  double _resample_rate;  //! Actual Resample Rate
  int _delta_samples;     //! Filter Skip
  int _last_filter;       //! Last filter used during filter execution
  double _accum;          //! Accumulated float filter skips
  double _flt_rate;       //! Actual Decimation Rate
  int _num_samples_kernel_out;  //! Length of the Kernel channel out buffer
  int _num_samples_kernel_in;   //! Length of the Kernel channel in buffer
  int _shared_mem_size;         //! Size of the shared memory
  int _num_taps_per_filter;     //! Numbers of taps per filter in the filterbank
  std::vector<float> _prot_taps;  //! Vector with prototype filter taps
  std::vector<float> _poly_taps;  //! Vector with polyphase filter taps
  std::vector<float> _diff_taps;  //! Vector with polyphase differential taps

  // CUDA Grid Variables
  enum { warp_size = 32 };    //! Compile time constant warp size
  int _num_channels;          //! Number of channels to process
  int _num_threadblocks;      //! Number of threadblocks in grid
  int _num_threads_in_block;  //! Number of threads in block
  int _cuda_shared_mem_size;  //! Size of the shared memory
  dim3 _filter_block_config;  //! CUDA Block onfig for the filter
  cuda_config _grid_config;   //! CUDA Grid config for the filter

  // CUDA GPU_BUFFERS
  int _cuda_buffer_len;            //! Size of the CUDA Channel Buffer
  cafe::cuda_unique_ptr _history;  //! Pointer to allocated history buffer
                                   //! on the device
  pfb_cuda_config _cuda_config;    //! Complete Cuda Config

 public:
  //! Typedef for the shared ptr of the pfb_arb_resampler class
  typedef std::shared_ptr<pfb_arb_resampler> pfb_arb_resampler_sptr;

  /*****************************************************************************/ /*!
   * @brief Constructor for the pfb_arb_resampler class
   * @param in std::vector<float> &taps Vector with prototype filter taps
   * @param in int    num_filters   Number of filters/Interpolation rate
   * @param in double resample_rate The Actual resample rate
   * @param in int    num_channels  Number of channels to be processed
   ***************************************************************************/
  pfb_arb_resampler(const std::vector<float> &taps, int num_filters,
                    double resample_rate, int num_channels);

  /*****************************************************************************/ /*!
   * @brief Destructor for the pfb_arb_resampler class
   ***************************************************************************/
  ~pfb_arb_resampler();

  /*****************************************************************************/ /*!
   * @brief Factory function for the pfb_arb_resampler class
   * @param in std::vector<float> &taps Vector with prototype filter taps
   * @param in int    num_filters   Number of filters/Interpolation rate
   * @param in double resample_rate The Actual resample rate
   * @param in int    num_channels  Number of channels to be processed
   ***************************************************************************/
  static pfb_arb_resampler_sptr pfb_arb_resampler_factory(
      const std::vector<float> &taps, int num_filters, double resample_rate,
      int num_channels);

  /*****************************************************************************/ /*!
   * @brief Sets up the taps to be used
   * @param in std::vector<float> &prot_taps Vector with prototype filter taps
   * @param in std::vector<float> &poly_taps Vector with pfb filter taps
   * @param in int    num_filters   Number of filters/Interpolation rate
   ***************************************************************************/
  void set_taps(const std::vector<float> &prot_taps,
                std::vector<float> &poly_taps, int num_filters);

  /*****************************************************************************/ /*!
   * @brief Sets up the differential taps to be used
   * @param in std::vector<float> &taps Vector with prototype filter taps
   * @param in std::vector<float> &diff_taps Vector with differential filter
   *                                         taps
   * @param in int    num_filters   Number of filters/Interpolation rate
   ***************************************************************************/
  void set_diff_taps(const std::vector<float> &taps,
                     std::vector<float> &difftaps);

  /*****************************************************************************/ /*!
   * @brief Wrapper function to set all taps
   ***************************************************************************/
  void set_filters();

  /*****************************************************************************/ /*!
   * @brief Wrapper function to setup the CUDA Grid
   ***************************************************************************/
  void set_grid();

  /*****************************************************************************/ /*!
   * @brief Parameter overloaded Wrapper function to setup the CUDA Grid
   * @param in int  block_dimx X Dimension of the CUDA Block
   * @param in int  block_dimy Y Dimension of the CUDA Block
   * @param in dim3 grid_dim   Dimension of the CUDA Grid
   ***************************************************************************/
  void set_grid(int block_dimx, int block_dimy, dim3 grid_dim);

  /*****************************************************************************/ /*!
   * @brief Sets the filter and input skips used
   * @param in double  rate Resample Rate used
   * @param in int  num_filters Number of filter in the pfb
   ***************************************************************************/
  void set_indexing(double rate, int num_filters);

  /*****************************************************************************/ /*!
   * @brief Generates a plan for the filter and input skips used
   * @param in int  len_inbuffer Number of samples at the input
   ***************************************************************************/
  int gen_index_plan(int len_inbuffer);

  /*****************************************************************************/ /*!
   * @brief Sets all the local variable in constant memory on the device
   ***************************************************************************/
  void set_constant_symbols();

  /*****************************************************************************/ /*!
   * @brief Sets the filter tail on the device
   ***************************************************************************/
  void set_history();

  /*****************************************************************************/ /*!
   * @brief Sets the the size of a single channel out buffer on the device
   * @param in int num_samples Number of samples in a channel out buffer
   ***************************************************************************/
  void set_num_samples_gpu(int num_samples);

  /*****************************************************************************/ /*!
   * @brief Sets the the CUDA block config on the device
   * @param in unsigned int block_size The number of blocks in a grid
   ***************************************************************************/
  int set_filter_block_config(unsigned int block_size);

  /*****************************************************************************/ /*!
   * @brief Executes the CUDA Arb Resampler kernel on the device
   * @param in int num_insamples Numbers of samples to be processed
   * @param in float2 *input_buffer Pointer to input buffer on device
   *                                Global Memory
   * @param in float2 *output_buffer Pointer to output buffer on device
   *                                Global Memory
   ***************************************************************************/
  int filter(int num_insamples, float2 *input_buffer, float2 *output_buffer);
};

} /* namespace filter */

#endif /* __PFB_ARB_RESAMPLER_H__ */
