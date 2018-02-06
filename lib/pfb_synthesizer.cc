#include "pfb_synthesizer.h"
#include <filterbank_execute.h>
#include <pfb.h>
#include <iostream>
#include "stdlib.h"
#include "string.h"

namespace pacs
{
namespace filter
{
pfbSynthesizer_ccf::pfbSynthesizer_ccf(const vector_float &taps,
                                       int numChannels, int packetSize,
                                       int numReplicas)
    : _numChannels(numChannels),
      _numTimeSlots(90 / numChannels),
      _numReplicas(numReplicas),
      _packetSize(packetSize),
      _cudaBufferLen(0),
      _cudaSharedMemSize(0),
      _burstBuffer(nullptr),
      _gInStreams(nullptr, free),
      _gFFTInBuffer(nullptr, free),
      _gFFTOutBuffer(nullptr, free),
      _gFilterInBuffer(nullptr, free),
      _gFilterOutBuffer(nullptr, free)
{
  int err = 0;
  err = cuda::pacs_init_cuda();
  if (err != 0) {
    throw std::runtime_error("Error: Could not initialize GPU!");
  }

  setFilters(taps, numChannels);
  setGrid();
  setStreams();
  setConstantSymbols();

  _slots.push_back(57);
  _slots.push_back(57);
  _slots.push_back(57);

  // Allocate the burst buffer

  cudaMallocHost((void **)&_burstBuffer,
                 sizeof(complex_float) * _cudaBufferLen);
  memset(_burstBuffer, 0, sizeof(complex_float) * _cudaBufferLen);

  int outSize[1] = {_numChannels};

  cufftPlanMany(&_fftPlan, 1, &numChannels, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C,
                128);
}

pfbSynthesizer_ccf::~pfbSynthesizer_ccf()
{
  cudaFree(_burstBuffer);
  cufftDestroy(_fftPlan);
}

void pfbSynthesizer_ccf::setFilters(const vector_float taps,
                                    unsigned int numChannels)
{
  // We want to split the filter in M=numChannels polyphase partitions
  // So we calculate how many taps we need per filter and round that up to
  // the next integer
  int nTapsPerFilter = (int)ceil(1.0 * taps.size() / numChannels);

  // Resize the filter so it can accomodate the numChannels*nTapsPerFilter new
  // taps
  // The newly added taps are zeroed out
  vector_float tapsExpanded = taps;
  while (tapsExpanded.size() < nTapsPerFilter * numChannels) {
    tapsExpanded.push_back(0.0);
  }
  _pTaps = vector_float(nTapsPerFilter * numChannels, 0);

  for (int i = 0; i < numChannels; ++i) {
    for (int k = 0; k < nTapsPerFilter; ++k) {
      _pTaps[i * nTapsPerFilter + k] = tapsExpanded[k * numChannels + i];
    }
  }

  _nTapsPerFilter = nTapsPerFilter;
}

void pfbSynthesizer_ccf::setStreams()
{
  // Reset the pointers
  _gInStreams.reset();
  _gFFTInBuffer.reset();
  _gFilterInBuffer.reset();
  _gFilterOutBuffer.reset();
  _gFilterOutBuffer.reset();

  // allocate the input stream
  _gInStreams = cuda::create_cuda_unique_ptr(_cudaBufferLen * sizeof(float2));
  _gFFTInBuffer = cuda::create_cuda_unique_ptr(_cudaBufferLen * sizeof(float2));
  _gFFTOutBuffer =
      cuda::create_cuda_unique_ptr(_cudaBufferLen * sizeof(float2));
  _gFilterInBuffer =
      cuda::create_cuda_unique_ptr(_cudaBufferLen * sizeof(float2));
  _gFilterOutBuffer =
      cuda::create_cuda_unique_ptr(_cudaBufferLen * sizeof(float2));
}

void pfbSynthesizer_ccf::setGrid()
{
  int filterHistory = _nTapsPerFilter - 1;
  int samplesPerStream = (_packetSize * _numTimeSlots + 2 * filterHistory);
  int expandedStream = 128;
  int expandedGrid = ceil((1.0 * expandedStream) / 16);
  std::cout << expandedGrid << "\n";

  _cuda_config.shuffleGridDim = _numChannels;
  _cuda_config.shuffleBlockDimX = expandedStream;
  _cuda_config.shuffleBlockDimY = 1;

  _cuda_config.fbBlockDimY = 1;
  _cuda_config.fbBlockDimX = _packetSize * _numTimeSlots + filterHistory;
  _cuda_config.fbGridDim.x = _numChannels;
  _cuda_config.fbGridDim.y = 1;
  _cuda_config.fbGridDim.z = 1;

  _cudaBufferLen = _numChannels * expandedStream;
  _cudaSharedMemSize =
      (_cuda_config.fbBlockDimX + filterHistory + 1) * sizeof(float2);

  _fsGridDim = 8;
  _fsBlockDimX = _numChannels;
  _fsBlockDimY = 16;
}

void pfbSynthesizer_ccf::setConstantSymbols()
{
  auto *cfg = &_cuda_config;

  // This is the number of filtertaps or in the PFB banks, the input samples are
  // shifted
  // For the fully decimated case this is equal to the number of channels
  int sample_shift = 0;
  int stream_len = _cudaBufferLen / cfg->fbGridDim.x;

#if DEBUG_PRINT
  std::cout << "Length of one stream buffer is set to " << (int)stream_len
            << " entries\n";
#endif

  int prot_filter_size = _pTaps.size();
  int taps_per_channel = _pTaps.size() / _numChannels;

  int cuda_err = 0;
  cuda_err = set_filterbank_constants(&stream_len, &sample_shift, &_pTaps[0],
                                      &taps_per_channel, prot_filter_size);

  if (cuda_err)
    throw std::runtime_error("Error: Could not set constant memory on device");
}

int pfbSynthesizer_ccf::work(int numOutSamples, vector_int &numInSamples,
                             vector_const_void_star &inputSamples,
                             vector_void_star &outputSamples)
{
  complex_float *packet = (complex_float *)inputSamples[0];
  complex_float *burst = (complex_float *)outputSamples[0];
  memset(_burstBuffer, 0, _cudaBufferLen * sizeof(complex_float));

  for (int i = 0; i < _numReplicas; ++i) {
    // int filterHistoryOffset = (_slots[i]/_numTimeSlots + 1) *
    // (_nTapsPerFilter-1);
    // int bufferOffset = _slots[i] * 64 + _nTapsPerFilter-1 + (_slots[i]%1) *
    // _packetSize;
    int bufferOffset = 28 * 128 + _nTapsPerFilter - 1 + _packetSize;
    memcpy(&_burstBuffer[bufferOffset], packet,
           _packetSize * sizeof(complex_float));
  }

  cudaError_t err = cudaSuccess;
  err = cudaMemcpy(_gInStreams.get(), _burstBuffer, 128 * 45 * sizeof(float2),
                   cudaMemcpyHostToDevice);

  if (err != cudaSuccess) std::cout << "FUCK\n";

  cuda::shuffle_input(_gInStreams.get(), _gFFTInBuffer.get(), &_cuda_config);

  cufftExecC2C(_fftPlan, _gFFTInBuffer.get(), _gFFTOutBuffer.get(),
               CUFFT_INVERSE);

  cuda::synth_filter_shuffle(_gFFTOutBuffer.get(), _gFilterInBuffer.get(),
                             _fsGridDim, _fsBlockDimX, _fsBlockDimY);

  cuda::anti_imaging_filter(_gFilterInBuffer.get(), _gFilterOutBuffer.get(),
                            &_cuda_config, _cudaSharedMemSize);

  err = cudaMemcpy(burst, _gFilterOutBuffer.get(),
                   _numChannels *
                       (_packetSize * _numTimeSlots + _nTapsPerFilter - 1) *
                       sizeof(float2),
                   cudaMemcpyDeviceToHost);
  // err = cudaMemcpy(burst, _gFilterInBuffer.get(),
  // 128*45*sizeof(float2),cudaMemcpyDeviceToHost);

  if (err == 11) std::cout << "Fuck while copy back\n";

  return numOutSamples;
}
}
}
