#include "pfb_arb_resampler.h"
#include "arb_resampler.h"
#include "filterbank_execute.h"

#include <math.h>
#include <cmath>
#include <iostream>

namespace filter
{
pfb_arb_resampler::pfb_arb_resampler_sptr
pfb_arb_resampler::pfb_arb_resampler_factory(const std::vector<float> &taps,
                                             int num_filters,
                                             double resample_rate,
                                             int num_channels)
{
  return std::make_shared<pfb_arb_resampler>(taps, num_filters, resample_rate,
                                             num_channels);
}

pfb_arb_resampler::pfb_arb_resampler(const std::vector<float> &taps,
                                     int num_filters, double resample_rate,
                                     int num_channels)
: _num_filters(num_filters),
  _resample_rate(resample_rate),
  _prot_taps(taps),
  _num_channels(num_channels),
  _history(nullptr, free)
{

  _last_filter = (taps.size() / 2) % num_filters;
  set_filters();
  set_indexing(_resample_rate, num_filters);
  set_grid();
  set_history();

  try {
    set_constant_symbols();
  } catch (std::exception &e) {
    throw;
  }
}

/*****************************************************************************/
pfb_arb_resampler::~pfb_arb_resampler() {}

/*****************************************************************************/
void pfb_arb_resampler::set_taps(const std::vector<float> &prot_taps,
                                 std::vector<float> &poly_taps, int num_filters)
{
  unsigned int len_prot_filter = prot_taps.size();
  unsigned int len_poly_filter =
      std::ceil((float)len_prot_filter / num_filters);
  unsigned int len_prot_filter_pad = len_poly_filter * num_filters;

  // std::cout << "Length of Protoptype Filter : " << len_prot_filter << "\n";
  // std::cout << "Length of Protoptype Filter (padded) : " <<
  // len_prot_filter_pad << "\n";
  // std::cout << "Length of Polyphase Filter Component : " << len_poly_filter
  // << "\n";

  std::vector<float> temp_taps = prot_taps;

  poly_taps.resize(len_prot_filter_pad, 0.0);
  temp_taps.resize(len_prot_filter_pad, 0.0);

  for (unsigned int i = 0; i < len_poly_filter; ++i) {
    for (int j = 0; j < num_filters; ++j) {
      poly_taps[len_poly_filter * j + i] = temp_taps[i * num_filters + j];
    }
  }

  _num_taps_per_filter = (int)(len_poly_filter + 0.5);
  //std::cout << "We're having " << _num_taps_per_filter << " taps per filter\n";
}

/*****************************************************************************/
void pfb_arb_resampler::set_diff_taps(const std::vector<float> &taps,
                                      std::vector<float> &difftaps)
{
  std::vector<float> diff_coeffs{-1, 1};

  // Make sure that the difftaps always have the right size
  // So we start with zero!
  difftaps.resize(0);

  for (unsigned int i = 0; i < taps.size() - 1; ++i) {
    float tap = 0;
    for (unsigned int j = 0; j < diff_coeffs.size(); ++j) {
      tap += diff_coeffs[j] * taps[i + j];
    }
    difftaps.push_back(tap);
  }

  // Last one is zero because we cannot have a linear interpolation between two
  // neighbouring points there
  difftaps.push_back(0);
}

void pfb_arb_resampler::set_filters()
{
  std::vector<float> diff_taps;
  set_diff_taps(_prot_taps, diff_taps);
  set_taps(_prot_taps, _poly_taps, _num_filters);
  set_taps(diff_taps, _diff_taps, _num_filters);
}

/*****************************************************************************/
void pfb_arb_resampler::set_grid()
{
  switch (_num_channels) {
    case 45:
      _num_samples_kernel_out = 256;
      _shared_mem_size = _num_samples_kernel_out + _num_taps_per_filter;
      break;
    case 30:
      _num_samples_kernel_out = 256;
      _shared_mem_size = _num_samples_kernel_out + _num_taps_per_filter;
    default:
      _num_samples_kernel_out = 256;
      _shared_mem_size = _num_samples_kernel_out + _num_taps_per_filter;
  }

  _shared_mem_size *= sizeof(float2);

  // dim3 grid_dim = dim3(_num_channels,1,1);
  // set_grid(_num_samples_kernel_out, 1, grid_dim);
  _cuda_buffer_len =
      (int)std::ceil(_num_samples_kernel_out * _resample_rate) * _num_channels;

  // std::cout << "Resample Rate " << _resample_rate << "\n";
  // std::cout << "Cuda Buffer Len " << _cuda_buffer_len << "\n";
}

/*****************************************************************************/
void pfb_arb_resampler::set_grid(int block_dimx, int block_dimy, dim3 grid_dim)
{
  _grid_config.rs_block_dimx = block_dimx;
  _grid_config.rs_block_dimy = block_dimy;
  _grid_config.rs_grid_dim = grid_dim;
}

/*****************************************************************************/
void pfb_arb_resampler::set_indexing(double rate, int num_filters)
{
  _int_rate = num_filters;
  _delta = _int_rate / rate;
  _accum = 0;
  _flt_rate = _delta - std::floor(_delta);

  std::cout << "rate      = " << rate << std::endl;
  std::cout << "_int_rate = " << _int_rate << std::endl;
  std::cout << "_flt_rate = " << _flt_rate << std::endl;
  std::cout << "_delta    = " << _delta << std::endl;
}

/*****************************************************************************/
int pfb_arb_resampler::gen_index_plan(int len_inbuffer)
{
  int predicted_ops = 0;

  // check at which filter we will end up after len_inbuffer samples
  double end_filter = len_inbuffer * _num_filters;
  // remove the accumulated offset
  end_filter -= (_last_filter + _accum);

  // Caclulate the number of filtering operations to do that
  predicted_ops = (int)std::ceil(end_filter / _delta);
  _last_filter =
      (int)fmodf((predicted_ops + 1) * _delta + _last_filter, _num_filters);

  // Store the newly accumulated offet
  _accum = fmodf(predicted_ops * _flt_rate + _accum, 1.0);

  return predicted_ops;
}

/*****************************************************************************/
void pfb_arb_resampler::set_constant_symbols()
{
  int num_taps = _poly_taps.size() / _num_filters;
  int res = set_resampler_constants(
      &_delta, &_accum, &_flt_rate, &_num_filters, &_last_filter, &num_taps,
      _poly_taps.data(), _diff_taps.data(), &_num_samples_kernel_out);
  if (res) std::cout << "Error: Could not copy constant symbols\n";
}

void pfb_arb_resampler::set_history()
{
  _history.reset();
  size_t size = _num_filtes * (_num_taps_per_filter - 1);
  _history = cafe::create_cuda_unique_ptr(size * sizeof(float2));
  cudaMemset((void *)_history.get(), 0, size * sizeof(float2));
}

/*****************************************************************************/
void pfb_arb_resampler::set_num_samples_gpu(int num_samples)
{
  int res = set_num_samples(&num_samples);
  if (res) {
    std::cout << "Error, could not update num_samples on the GPU\n";
    return;
  }
}

/*****************************************************************************/
int pfb_arb_resampler::set_filter_block_config(unsigned int block_size)
{
  const unsigned int max_x_size = 256;
  const unsigned int max_kernel = 1024;
  unsigned int kernel_in = block_size;
  unsigned int kernel_calls_p_block = 1;
  unsigned int new_block =
      static_cast<int>(std::ceil(kernel_in * _resample_rate));

  bool ready = false;

  while (!ready) {
    if (new_block > max_kernel) {
      kernel_in /= 2;
      new_block =
          static_cast<unsigned int>(std::ceil(kernel_in * _resample_rate));
      kernel_calls_p_block++;
      continue;
    }

    unsigned int kernel_block = new_block;
    kernel_block =
        static_cast<unsigned int>(std::ceil(kernel_block / 32.0)) * 32;
    std::cout << kernel_calls_p_block << " kernel calls on " << kernel_block
              << " samples\n";

    if (kernel_block % 32) {
      std::cout << "Block Size must be multiple of 32\n";
      return -1;
    }

    unsigned int x_dim = 0;
    unsigned int y_dim = 0;

    unsigned int mod = kernel_block % max_x_size;
    if (!mod) {
      x_dim = max_x_size;
      y_dim = kernel_block / max_x_size;
    } else {
      while (mod) {
        x_dim = mod;
        y_dim = kernel_block / x_dim;
        mod = kernel_block % mod;
      }
    }

    _num_samples_kernel_out = new_block;
    std::cout << "Filter Block configured to \n";
    std::cout << "Num Blocks: " << _num_channels << std::endl;
    std::cout << "Block Dim X " << x_dim << std::endl;
    std::cout << "Block Dim Y " << y_dim << std::endl;
    std::cout << "Kernel In Block " << kernel_in << std::endl;
    std::cout << "Samples produced by kernel: " << _num_samples_kernel_out
              << std::endl;
    _filter_block_config = dim3(x_dim, y_dim, 1);

    set_num_samples_gpu(block_size);
    int last_sample = static_cast<int>(kernel_in);
    if (set_last_sample(&last_sample)) {
      std::cout << "Could not set last sample on gpu\n";
    }

    _shared_mem_size = (block_size + _num_taps_per_filter + 1) * sizeof(float2);
    _num_samples_kernel_in = kernel_in;
    ready = true;
  }

  return 0;
}

/*****************************************************************************/
int pfb_arb_resampler::filter(int num_insamples, float2 *input_buffer,
                              float2 *output_buffer)
{
  int num_out_samples = 0;

  for (int i = 0; i < num_insamples;) {
    // Fire the kernel
    update_start_filter(&_last_filter);
    int num_ops = gen_index_plan(_num_samples_kernel_in);

    arb_resampler(input_buffer, output_buffer, _history.get(), _num_channels,
                  _filter_block_config, _shared_mem_size);

    i += _num_samples_kernel_in;
    num_out_samples += num_ops;
    input_buffer += _num_samples_kernel_in;
    output_buffer += num_ops;
  }

  return num_out_samples;
}

} /* namespace filter */
