#include "qa_PFBSynthesizer.h"
#include "pfb_synthesizer.h"
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

#define TESTING_PATH "/home/ghost/Development/pacs_esa_san/branches/jan_wip/filter-blocks/testing/"
#define TAP_FILE "../config/prot_filter_45x2.dat"

void
qa_PFBSynthesizer::t1()
{

  // Initialize the architecture of the host system
  arch_init();
  // Get the alignment for the system memory
  size_t alignment = get_alignment();
  // Create all the niffty helper variables
  const uint32_t numChannels = 45;
  const uint32_t numTimeSlots = 2;
  const int numReplicas = 3;
  const uint32_t numRuns = 10000;

  // Read the reference signals
  vector_float testTaps;
  readFloatSamples(TAP_FILE, testTaps);

  // Get the number of taps per filterbank
  int nTapsPerFilter = (int) ceil(1.0*testTaps.size()/numChannels);
  // Get the additional history to preload the filter taps
  int filter_hist = (nTapsPerFilter) * numChannels;
  // Magic values...¯\_(ツ)_/¯
  const uint32_t samplesPerPacket = 38;
  const uint32_t samplesPerBurst = numChannels*(samplesPerPacket*numTimeSlots + nTapsPerFilter-1);
  //const uint32_t samplesPerBurst = samplesPerPacket * numChannels * numTimeSlots;
  //const uint32_t samplesAfterFilter = samplesPerBurst + nTapsPerFilter - 1;       // We need to correctly flush the filter because we do not use a history

  complex_float *in;
  complex_float *out;

  // Allocate pinned memory on the host
  cudaMallocHost((void**)&in, sizeof(complex_float)*samplesPerPacket);
  cudaMallocHost((void**)&out, sizeof(complex_float)*samplesPerBurst);
  //cudaMallocHost((void**)&out, sizeof(complex_float)*128*45);

  memset(out, 0, samplesPerBurst*sizeof(complex_float));
  //memset(out, 0, 128*45*sizeof(complex_float));
  memset(in, 0, samplesPerPacket*sizeof(complex_float));

  // Set the noise stuff
  unsigned seed = 12345;
  std::default_random_engine generator(seed);

  std::normal_distribution<float> dist(0.0, 1.0);

  float *inFloat = (float*) in;

  for (int i = 0; i < samplesPerPacket; ++i) {
//   inFloat[2*i] = dist(generator);
//   inFloat[2*i+1] = dist(generator);
   inFloat[2*i] =   i;
   inFloat[2*i+1] = i;

  }

  //Setup Input Data
  //readComplexSamples(std::string("inSamples.dat"), inVec);

  vector_int numInSamples;
  numInSamples.push_back(samplesPerBurst);

  vector_const_void_star inSamples;
  inSamples.push_back(in);
  vector_void_star outSamples;
  // Setup the pointers to the correct Outputchannels
  outSamples.push_back(out);

  // Setup benchmarking
  std::chrono::nanoseconds ns;
  std::chrono::high_resolution_clock::time_point start;
  float average = 0;
  int64_t max = 0;
  int64_t min = INT64_MAX;

  pacs::filter::pfbSynthesizer_ccf fb(testTaps, numChannels, samplesPerPacket, numReplicas);


  for (uint32_t i = 0; i < numRuns; ++i) {

    start = std::chrono::high_resolution_clock::now();
    fb.work(samplesPerBurst, numInSamples,
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
  std::string filename("results/GPU_synth");
  filename = filename + ".dat";
  writeComplexSamplesToFile((complex_float*) outSamples[0], samplesPerBurst, filename);

  // Write the benchmark results to file
  // Remember to substract the filter history
  writeBenchmark(average, min, max, samplesPerBurst);

  cudaFreeHost(out);
  cudaFreeHost(in);

}
