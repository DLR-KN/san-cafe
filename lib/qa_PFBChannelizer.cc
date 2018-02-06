#include "qa_PFBChannelizer.h"
#include "pfb_channelizer.h"
#include "configure_pfb.h"
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

#include <gnuradio/types.h>
#include <volk/volk.h>
#include <gnuradio/filter/pfb_channelizer_ccf.h>

#define TESTING_PATH "/home/ghost/Development/pacs_esa_san/branches/jan_wip/filter-blocks/testing/"
#define CHANNEL_FILE "/home/pacs/pacs_dev/cuda-pfb/config/filter_test_input_45_2_3.txt"
#define TAP_FILE "../../config/steepest_taps_ever.dat"

void
qa_PFBChannelizer::t1()
{

  // Create all the niffty helper variables
  const uint32_t numChannels = 45;
  const uint32_t numRuns = 100000;
  const uint32_t oversampling = 3;

  // Read the reference signals
  vector_float testTaps;
  readFloatSamples(TAP_FILE, testTaps);

  vector_float resamplerTaps;
  std::string resamplerTapFile = ::pacs::cuda::get_config_dir();
  resamplerTapFile.append("arb_resampler_t1_taps.txt");
  readFloatSamples(resamplerTapFile, resamplerTaps);


  // Get the number of taps per filterbank
  int nTapsPerFilter = (int) ceil(1.0*testTaps.size()/numChannels);
  // Get the additional history to preload the filter taps
  int filter_hist = (nTapsPerFilter) * numChannels;
  // Magic values...¯\_(ツ)_/¯
  const uint32_t numSamples = 24480; //256*numChannels + filter_hist; //256*45
  const int samplesPerStream = 512;
  const int exceededSamples = 544;
  float resampling = 256.0/264;

  complex_float *in;
  complex_float *out;

  // Allocate pinned memory on the host
  cudaMallocHost((void**)&in, sizeof(complex_float)*numSamples);
  cudaMallocHost((void**)&out, sizeof(complex_float)*samplesPerStream*numChannels*oversampling);

  memset(out, 0, numSamples*sizeof(complex_float));
  memset(in, 0, numSamples*sizeof(complex_float));

  //Setup Input Data
  vector_complex_float inVec;
  //readComplexSamples(std::string("inSamples.dat"), inVec);
  std::string file = std::string(CHANNEL_FILE);
  readComplexSamples<float>(file, inVec);

  std::cout << inVec.size() << " Size of the invector\n";

  memcpy(in, &inVec[0], numSamples*sizeof(complex_float));

  vector_int numInSamples;
  numInSamples.push_back(numSamples);

  vector_const_void_star inSamples;
  inSamples.push_back(in);
  vector_void_star outSamples;
  // Setup the pointers to the correct Outputchannels
  for (int i = 0; i < numChannels; ++i) {

    outSamples.push_back(&out[i*samplesPerStream*oversampling]);

  }

  // Setup benchmarking
  std::chrono::nanoseconds ns;
  std::chrono::high_resolution_clock::time_point start;
  float average = 0;
  int64_t max = 0;
  int64_t min = INT64_MAX;

  pacs::filter::pfbChannelizer_ccf fb(testTaps, numChannels, 
                                      oversampling, resamplerTaps, resampling);


  for (uint32_t i = 0; i < numRuns; ++i) {

    start = std::chrono::high_resolution_clock::now();
    fb.work(numSamples, numInSamples,
            inSamples, outSamples);
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start);

    average += ((float)ns.count())/numRuns;
    if(ns.count() < min) min = ns.count();
    if(ns.count() > max) max = ns.count();

  }

  // Create the results directory
  struct stat st = {0};
  if(stat("resulst/", &st) == -1) {
    mkdir("results/", 0700);
  }

  // Write all Channels to file
  for (int i = 0; i < numChannels; ++i) {

    std::string filename("results/GPU_channel");
    filename = filename + std::to_string(i) + ".dat";
    writeComplexSamplesToFile((complex_float*) outSamples[i], samplesPerStream*oversampling, filename);

  }

  // Write the benchmark results to file
  // Remember to substract the filter history
  writeBenchmark(average, min, max, samplesPerStream*numChannels);

  cudaFreeHost(out);
  cudaFreeHost(in);

}

void
qa_PFBChannelizer::t2()
{
  // Create all the niffty helper variables
  const uint32_t numChannels = 45;
  const uint32_t oversampling = 3;
  int numRuns = 10000;
  float resampling = 256.0/264;

  // Read the reference signals
  vector_float testTaps;
  readFloatSamples(TAP_FILE, testTaps);

  // Resampler Taps
  vector_float resamplerTaps;
  std::string resamplerTapFile = ::pacs::cuda::get_config_dir();
  resamplerTapFile.append("arb_resampler_t1_taps.txt");
  readFloatSamples(resamplerTapFile, resamplerTaps);

  pacs::filter::pfbChannelizer_ccf pfb_pacs(testTaps, numChannels,
                                            oversampling, resamplerTaps, resampling);
  // Get the number of taps per filterbank
  // Magic values...¯\_(ツ)_/¯

  complex_float *in;
  complex_float *in_gr;
  complex_float *out_pacs;
  complex_float *out_gr;

  //Setup Input Data
  std::vector<std::complex<float>> inVec;
  //readComplexSamples(std::string("inSamples.dat"), inVec);
  std::string file = std::string(CHANNEL_FILE);
  readComplexSamples<float>(file, inVec);
  const uint32_t numSamples = inVec.size();
  const uint32_t samplesPerStream = inVec.size() / numChannels;
  const uint32_t numTaps = std::ceil(testTaps.size() / numChannels) * numChannels;
  const uint32_t exceededSamples = pfb_pacs.getInMultiple() / numChannels;
  const uint32_t totalSamples = exceededSamples * numChannels;
  const uint32_t history = totalSamples - numSamples;

  // Allocate pinned memory on the host
  cudaMallocHost((void**)&in, sizeof(complex_float)*totalSamples);
  cudaMallocHost((void**)&out_pacs, sizeof(complex_float)*totalSamples*oversampling);

  out_gr = (complex_float*) malloc(totalSamples*oversampling*sizeof(complex_float));
  in_gr = (complex_float*) malloc(totalSamples*sizeof(complex_float));

  memset(out_gr, 0, numSamples*oversampling*sizeof(complex_float));
  memset(out_pacs, 0, numSamples*oversampling*sizeof(complex_float));
  memset(in, 0, totalSamples*sizeof(complex_float));
  memset(in_gr, 0, totalSamples*sizeof(complex_float));

  memcpy(in, &inVec[history], numSamples*sizeof(complex_float));

  vector_int numInSamples;
  numInSamples.push_back(totalSamples);

  vector_int numInSamples_gr;
  numInSamples_gr.push_back(totalSamples);

  vector_const_void_star inSamples;
  inSamples.push_back(in);

  vector_const_void_star inSamples_gr;
  for (int i = 0; i < numChannels; ++i) {
    for (int j = 0; j < exceededSamples; ++j) {
      in_gr[i*exceededSamples+j] = in[i+j*numChannels];
    }
    inSamples_gr.push_back(&in_gr[i*exceededSamples]);
  }

  vector_void_star outSamples_pacs;
  vector_void_star outSamples_gr;

  // Setup the pointers to the correct Outputchannels
  for (int i = 0; i < numChannels; ++i) {

    outSamples_pacs.push_back(&out_pacs[i*samplesPerStream*oversampling]);
    outSamples_gr.push_back(&out_gr[i*samplesPerStream*oversampling]);

  }



  // Setup benchmarking
  std::chrono::nanoseconds ns;
  std::chrono::nanoseconds ns_gr;
  std::chrono::high_resolution_clock::time_point start;
  float average = 0;
  int64_t max = 0;
  int64_t min = INT64_MAX;

  float average_gr = 0;
  int64_t max_gr = 0;
  int64_t min_gr = INT64_MAX;

  for (int i = 0; i < numRuns; ++i) {

    gr::filter::pfb_channelizer_ccf::sptr pfb_gr = gr::filter::pfb_channelizer_ccf::make(
                                                  numChannels, testTaps, oversampling);
    start = std::chrono::high_resolution_clock::now();
    pfb_pacs.work(numSamples, numInSamples,
            inSamples, outSamples_pacs);
    ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start);

    start = std::chrono::high_resolution_clock::now();
    pfb_gr->general_work(samplesPerStream*oversampling, numInSamples_gr, inSamples_gr, outSamples_gr);
    ns_gr = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start);

    average += ((float)ns.count())/numRuns;
    if(ns.count() < min) min = ns.count();
    if(ns.count() > max) max = ns.count();

    average_gr += ((float)ns_gr.count())/numRuns;
    if(ns_gr.count() < min_gr) min_gr = ns_gr.count();
    if(ns_gr.count() > max_gr) max_gr = ns_gr.count();

    for (int i = 0*samplesPerStream*oversampling; i < samplesPerStream*oversampling; ++i)
    {

      //std::cout << "GNU Radio Samples = " << out_gr[i] << " PACS Samples " << out_pacs[i] << " at " << i<<"\n";
      //ASSERT_FLOAT_ALMOST_EQUAL(out_gr[i].real(), out_pacs[i].real(), 1e-4);
      //ASSERT_FLOAT_ALMOST_EQUAL(out_gr[i].imag(), out_pacs[i].imag(), 1e-4);
    }

  }

    std::cout << "Benchmark for GNU Radio:\n";
    writeBenchmark(average_gr, min_gr, max_gr, samplesPerStream*numChannels);
    std::cout << "\n\n\nBenchmark for PACS:\n";
    writeBenchmark(average, min, max, samplesPerStream*numChannels);

}

/*****************************************************************************/
