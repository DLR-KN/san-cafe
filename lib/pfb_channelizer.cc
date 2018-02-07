#include <filterbank_execute.h>
#include <pfb.h>
#include <pfb_channelizer.h>
#include <string.h>
#include <iostream>

namespace filter
{
pfb_channelizer_ccf::pfb_channelizer_sptr
pfb_channelizer_ccf::pfb_channelizer_factory(const std::vector<float> &taps,
                                             int channels, int oversampling)
{
  return std::make_shared<pfb_channelizer_ccf>(taps, channels, oversampling);
}

pfb_channelizer_ccf::pfb_channelizer_ccf(const std::vector<float> &taps,
                                         int channels, int oversampling)
: _num_channels(channels),
  _oversampling(oversampling),
  _num_thread_blocks(channels),
  _samples_to_process(0),
  _g_instream(nullptr, free),
  _g_instreams(nullptr, free),
  _fft_inbuffer(nullptr, free),
  _fft_outbuffer(nullptr, free)
{
  int err = 0;
  err = cafe::cafe_init_cuda();
  if (err != 0) {
    throw std::runtime_error(std::string("Error: Could not initialize GPU"));
  }

  // Set GPU specific stuff
  set_filters(taps, channels);
  set_grid();

  // Set the relevant variables for the scheduler regarding in/output batch
  // sizes
  _pfb_produce = _samples_to_process * _oversampling;

  set_streams();

  // Set the all the constant variable on the GPU
  //  If this does not suceed rethrow
  try {
    set_constant_symbols();
  } catch (std::exception &e) {
    throw;
  }

  int inembed = _pfb_produce * channels;
  int onembed = _pfb_produce * channels;
  cufftResult res =
      cufftPlanMany(&_fft_plan, 1, &channels, &inembed, 1, channels, &onembed,
                    _pfb_produce, 1, CUFFT_C2C, _pfb_produce);

  if (res != CUFFT_SUCCESS) {
    std::cout << "Error: could not set plan for CUFFT with err " << res
              << std::endl;
  }

  // Setup the output buffer
  cudaError cuda_err =
      cudaMallocHost((void **)&_fft_out,
                     sizeof(std::complex<float>) * channels * _pfb_produce);
  if (cuda_err) {
    std::cout
        << "Error: Could not allocate memory for the output buffer, error : "
        << cuda_err << std::endl;
  }
  std::cout << "Finished setting up the Polyphase Filterbank\n";
}

/*****************************************************************************/
pfb_channelizer_ccf::~pfb_channelizer_ccf()
{
  cufftDestroy(_fft_plan);

  if (_fft_out != nullptr) {
    free(_fft_out);
    _fft_out = nullptr;
  }
}

void pfb_channelizer_ccf::set_filters(const std::vector<float> &taps,
                                      int num_channels)
{
  // We want to split the filter in M=num_channels polyphase partitions
  // So we calculate how many taps we need per filter and round that up to
  // the next integer
  int num_taps_per_filter = (int)ceil(1.0 * taps.size() / num_channels);

  // Resize the filter so it can accomodate the num_channels*nTapsPerFilter new
  // taps
  // The newly added taps are zeroed out
  std::vector<float> taps_expanded = taps;
  while (taps_expanded.size() <
         static_cast<unsigned int>(num_taps_per_filter * num_channels)) {
    taps_expanded.push_back(0.0);
  }
  _p_taps = std::vector<float>(num_taps_per_filter * num_channels, 0);

  for (int i = 0; i < num_channels; ++i) {
    for (int k = 0; k < num_taps_per_filter; ++k) {
      _p_taps[i * num_taps_per_filter + k] =
          taps_expanded[k * num_channels + i];
    }
  }

  _num_taps_per_filter = num_taps_per_filter;
  std::cout << _num_taps_per_filter << " taps per filter with " << taps.size()
            << " taps overall\n";
}

void pfb_channelizer_ccf::set_streams()
{
  // Reset the pointers
  _g_instream.reset();
  _g_instreams.reset();
  _fft_inbuffer.reset();
  _fft_outbuffer.reset();

  // allocate the input stream
  _g_instream =
      cafe::create_cuda_unique_ptr(_cuda_buffer_len * sizeof(float) * 2);
  _g_instreams =
      cafe::create_cuda_unique_ptr(_cuda_buffer_len * sizeof(float) * 2);
  _fft_inbuffer = cafe::create_cuda_unique_ptr(
      _num_channels * (_pfb_produce + 2) * sizeof(float2));
  _fft_outbuffer = cafe::create_cuda_unique_ptr(
      _num_channels * (_pfb_produce + 2) * sizeof(float2));
}

void pfb_channelizer_ccf::set_grid()
{
  // So we want enough threads per Block to somehow saturate the cores and the
  // pipeline
  // of our Device. But we also need to take care of memory consumption in
  // global and
  // shared memory. This tries at least to accomodate for different oversampling
  // factors
  // We also want to have a threads_per_block count that is a multiple of a warp
  // (32) to
  // optimize memory access and indexing in our kernel
  int samples_per_tb = 1;
  int sh_grid_mulitplicator = 1;
  if (_oversampling <= 4) {
    samples_per_tb = 64;
    sh_grid_mulitplicator = 2;
  } else if (_oversampling <= 8) {
    samples_per_tb = 64;
    sh_grid_mulitplicator = 2;
  } else {
    samples_per_tb = 32;
    sh_grid_mulitplicator = 1;
  }
  _threads_per_block = _oversampling * samples_per_tb;

  _cuda_config.fb_blockdim_y = _oversampling;
  _cuda_config.fb_blockdim_x = samples_per_tb;
  _cuda_config.fb_griddim.x = _num_channels;
  _cuda_config.fb_griddim.y = default_fb_grid_dim_y;
  _cuda_config.fb_griddim.z = 1;

  // Expand the grid, so we shuffle more input sample than we process
  // in the filterbank. But this ensures an even grid distribution
  // int filter_hist = (_nTapsPerFilter) * _num_channels;
  int additional_blocks =
      ceil(1.0 * (_num_taps_per_filter - 1) / default_sh_grid_dim);
  int expanded_grid =
      additional_blocks + default_sh_grid_dim * sh_grid_mulitplicator;

  //#if DEBUG_PRINT
  std::cout << "Expanded Grid set to " << expanded_grid << " entries\n";
  //#endif

  _cuda_config.shuffle_griddim = expanded_grid;
  _cuda_config.shuffle_blockdim_y = 16;
  _cuda_config.shuffle_blockdim_x = _num_channels;

  // int gridExcess = additional_blocks * _cuda_config.shuffle_blockdim_y;
  // int gridOverhead = gridExcess - _nTapsPerFilter + 1;
  // setExcess(gridOverhead*_num_channels);
  // Set Sizes for the GPU Buffers
  _samples_to_process = _cuda_config.fb_blockdim_x * _cuda_config.fb_griddim.y;
  _cuda_shared_mem_size =
      (samples_per_tb + _num_taps_per_filter) * sizeof(float) * 2;
  _cuda_buffer_len =
      (expanded_grid * _cuda_config.shuffle_blockdim_y) * _num_channels;
}

void pfb_channelizer_ccf::set_constant_symbols()
{
  auto *cfg = &_cuda_config;

  // This is the number of filtertaps or in the PFB banks, the input samples are
  // shifted
  // For the fully decimated case this is equal to the number of channels
  int sample_shift = cfg->fb_griddim.x / cfg->fb_blockdim_y;
  int stream_len = _cuda_buffer_len / cfg->fb_griddim.x;

#if DEBUG_PRINT
  std::cout << "Length of one stream buffer is set to " << (int)stream_len
            << " entries\n";
#endif

  int prot_filter_size = _p_taps.size();
  int taps_per_channel = _p_taps.size() / _num_channels;

  int cuda_err = 0;
  cuda_err = set_filterbank_constants(&stream_len, &sample_shift, &_p_taps[0],
                                      &taps_per_channel, prot_filter_size);

  if (cuda_err)
    throw std::runtime_error("Error: Could not set constant memory on device");
}

void pfb_channelizer_ccf::print_taps()
{
  std::cout << "\nTaps: \n";
  for (unsigned int i = 0; i < _p_taps.size(); ++i) {
    std::cout << "Tap[" << i << "] = " << _p_taps[i] << "\n";
  }
  std::cout << "\n\n";
}

int pfb_channelizer_ccf::filter(std::complex<float> *input,
                                std::complex<float> *output,
                                unsigned int num_samples)
{
  const int num_chunks = (const int)std::floor(num_samples / _cuda_buffer_len);
  unsigned int produced = 0;

  for (int i = 0; i < num_chunks; ++i) {
    // Copy the input data to the device
    cudaError err =
        cudaMemcpy(_g_instream.get(), input, _cuda_buffer_len * sizeof(float2),
                   cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
      std::cout << "Error copying input to GPU\n";
    }

    // Execute the PFB operation
    cuda::pfb_execute(_g_instream.get(), _g_instreams.get(),
                      _fft_inbuffer.get(), _fft_outbuffer.get(), &_cuda_config,
                      _cuda_shared_mem_size);

    // Despin the samples via the CUDA FFT
    int ret = cufftExecC2C(_fft_plan, _fft_inbuffer.get(), _fft_outbuffer.get(),
                           CUFFT_INVERSE);
    if (ret) {
      return 0;
    }

    // Copy the samples back to host
    cudaMemcpy(_fft_out, _fft_outbuffer.get(),
               _pfb_produce * _num_channels * sizeof(float2),
               cudaMemcpyDeviceToHost);

    // Shuffle the data to the correct channel output buffer
    // for (int k = 0; k < _num_channels; ++k) {
    //    out = (complex_float*) outputSamples[k];
    //
    //    for (int j = 0; j < resamplerOut; ++j) {
    //      out[j+produced] = _resamplerOut[k*resamplerOut+j];
    //    }
    // }

    produced += _pfb_produce;
  }

  return produced;
}

} /* Namespace filter */
