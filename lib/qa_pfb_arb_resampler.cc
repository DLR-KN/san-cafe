#include "qa_pfb_arb_resampler.h"
#include "pfb_arb_resampler.h"
#include "pacs_cuda_helper.h"
#include "types.h"
#include "readSamples.h"
#include "benchmarks.h"
#include "signalGenerator.h"
#include "writeSamplesToFile.h"

#include <iostream>
#include <chrono>
#include <random>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

void
qa_pfb_arb_resampler::t1()
{
  namespace pf =  pacs::filter;

  // init the Device
  int err = pacs::cuda::pacs_init_cuda();
  if(err != 0) {
    throw std::runtime_error(std::string("Error: Could not initialize GPU"));
  }

  const uint32_t num_channels = 45;
  const uint32_t num_runs = 2;
  const uint32_t num_filter = 32;
  const uint32_t num_samples = 512;
  uint32_t total_samples = num_samples * num_channels;
  double sample_rate = 256/273.0;

  vector_float taps;

  std::string TAP_FILE = pacs::cuda::get_config_dir();
  TAP_FILE.append("arb_resampler_t1_taps.txt");
  readFloatSamples(TAP_FILE, taps);

  int taps_per_filter = std::ceil(taps.size()/num_filter);

  int max_outsamples = std::ceil(num_samples*sample_rate)*num_channels;

  // Setup the Device Buffer
  pacs::cuda::cuda_unique_ptr g_inbuffer(nullptr, free);
  pacs::cuda::cuda_unique_ptr g_outbuffer(nullptr, free);
  g_inbuffer.reset();
  g_outbuffer.reset();
  g_inbuffer = pacs::cuda::create_cuda_unique_ptr(total_samples*sizeof(float2));
  g_outbuffer = pacs::cuda::create_cuda_unique_ptr(max_outsamples*sizeof(float2));

  pf::pfb_arb_resampler::pfb_arb_resampler_sptr resampler = pf::pfb_arb_resampler::pfb_arb_resampler_factory(taps, num_filter, sample_rate, num_channels);

  complex_float *in = (complex_float*) malloc(total_samples*sizeof(complex_float));
  complex_float *out = (complex_float*) malloc(total_samples*sizeof(complex_float));

  for (int i = 0; i < max_outsamples; ++i) {
    out[i] = complex_float(i, i+1);
  }
  memset(in, 0, total_samples*sizeof(complex_float));


  vector_complex_float file_samples;
  std::string CHANNEL_FILE = pacs::cuda::get_config_dir();
  CHANNEL_FILE.append("arb_samples.bin");
  std::string file = std::string(CHANNEL_FILE);
  readBinarySamples<complex_float>(file, file_samples);
  for (int i = 0; i < num_channels; ++i) {
    memcpy(&in[i*num_samples], file_samples.data(), num_samples*sizeof(complex_float));
  }

  // Setup benchmarking
  std::chrono::nanoseconds ns;
  std::chrono::high_resolution_clock::time_point start;
  float average = 0;
  int64_t max = 0;
  int64_t min = INT64_MAX;

  // Copy dato to GPU
  for (int i = 0; i < num_runs; ++i) {

    // Start benchmarking
    start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(g_inbuffer.get(), in, total_samples*sizeof(float2), cudaMemcpyHostToDevice);
    cudaError_t cuda_err = cudaGetLastError();
    if(cuda_err != cudaSuccess) {
      std::string error_str("Error: Could not copy data to device\n");
      std::cout << error_str;
      throw std::runtime_error(error_str);
    }

    // Resample the signal
    int samples_produced = resampler->filter(num_samples, g_inbuffer.get(), g_outbuffer.get());
    // Do the copying back from the device
    cudaMemcpy(out, g_outbuffer.get(), samples_produced*sizeof(float2), cudaMemcpyDeviceToHost);
    cuda_err = cudaGetLastError();
    if(cuda_err != cudaSuccess) {
      std::string error_str("Error: Could not copy data from device\n");
      std::cout << error_str;
      throw std::runtime_error(error_str);
    }
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start);

    average += ((float)ns.count())/num_runs;
    if(ns.count() < min) min = ns.count();
    if(ns.count() > max) max = ns.count();

  }

  // Write the benchmark results to file
  // Remember to substract the filter history
  writeBenchmark(average, min, max, 512);

  //writeComplexSamplesToFile(out, samples_produced, "pfb_arb_resampler_result.txt");

}
