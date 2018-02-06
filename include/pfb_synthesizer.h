#ifndef __PFB_SYNTHESIZER_H__
#define __PFB_SYNTHESIZER_H__
#include "memory"
#include "types.h"

#include "BasicBlock.h"
#include "arch_types.h"
#include "archs.h"

#include "cufft.h"
#include "pacs_cuda_helper.h"

namespace pacs {
namespace filter {

class pfbSynthesizer_ccf : public BasicBlock {
 private:
  enum { default_fb_grid_dim_y = 8 };
  enum { default_sh_grid_dim = 8 };

  cuda::pfb_cuda_config _cuda_config;

  vector_float _pTaps;
  int _numChannels;
  int _numTimeSlots;
  int _numReplicas;
  int _packetSize;
  int _nTapsPerFilter;
  int _cudaBufferLen;
  int _cudaSharedMemSize;

  // filter shuffle Variables
  int _fsGridDim;
  int _fsBlockDimX;
  int _fsBlockDimY;

  std::vector<int> _slots;

  // Grid Variables
  static const int _warpSize = 32;

  int _numThreadBlocks;
  int _threadsPerBlock;
  int _inBufferSize;

  complex_float *_burstBuffer;

  // CUDA GPU Buffers
  cuda::cuda_unique_ptr _gInStreams;
  cuda::cuda_unique_ptr _gFFTInBuffer;
  cuda::cuda_unique_ptr _gFFTOutBuffer;
  cuda::cuda_unique_ptr _gFilterInBuffer;
  cuda::cuda_unique_ptr _gFilterOutBuffer;

  cufftHandle _fftPlan;

 public:
  pfbSynthesizer_ccf(const vector_float &taps, int numChannels, int packetSize,
                     int numReplicas);
  ~pfbSynthesizer_ccf();

  void setFilters(const vector_float taps, unsigned int numChannels);

  void setConstantSymbols();

  void setStreams();

  void setGrid();

  // vector_float taps() const;

  int getChannels() { return _numChannels; }

  int work(int numOutSamples, vector_int &numInSamples,
           vector_const_void_star &inputSamples,
           vector_void_star &outputSamples);
};
}  // namespace filter
}  // namespace pacs

#endif /* ifndef __PFB_SYNTHESIZER_H__ */
