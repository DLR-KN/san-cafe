#include <gnuradio/types.h>
#include <volk/volk.h>
#include <gnuradio/filter/pfb_channelizer_ccf.h>

#include "readSamples.h"
#include "types.h"
#include "arch_types.h"
#include "archs.h"
#include "signalGenerator.h"
#include "benchmarks.h"
#include "writeSamplesToFile.h"

#include <iostream>
#include <chrono>
#include <cmath>
#include <time.h>

//#define TAP_FILE "/home/jkraemer/Development/DLR/cuda-pfb/config/prot_filter_45x2.dat"
#define TAP_FILE "../config/prot_filter_45x2.dat"

static void
mallocOutBufferVector(vector_void_star &outVec, uint32_t vecSize, 
                      uint32_t size,size_t alignment)
{
  complex_float *tmpPtr;
  for (unsigned int i = 0; i < vecSize; ++i) {

    tmpPtr = (complex_float*) arch_malloc(size*sizeof(complex_float), alignment);
    outVec.push_back(tmpPtr);

  }
  tmpPtr = nullptr;
}

static void
freeOutBufferVector(vector_void_star &outVec)
{
  for (auto i : outVec) {
    free(i);
  }
}

static void
condition_input(complex_float *in, int numChannels, int samplesPerChannel)
{
  int samples = numChannels*samplesPerChannel;
  complex_float *tmp;
  tmp = (complex_float*) malloc(samples*sizeof(complex_float));

  if(tmp == nullptr) {
    std::cout << "Error: Could not allocate temporary memory)\n";
    return;
  }

  memcpy(tmp, in, samples*sizeof(complex_float));
  memset(in, 0, samples*sizeof(complex_float));

  for (int i = 0; i < numChannels; ++i) {
    for (int j = 0; j < samplesPerChannel; ++j) {

      in[i*samplesPerChannel + j] = tmp[j*numChannels + i];

    }
  }

  free(tmp);

}


int main()
{
  arch_init();
  size_t alignment = get_alignment();

  const uint32_t numRuns = 1;
  const uint32_t numFilters = 45;
  const uint32_t numSamples = 256*numFilters; //365*45
  const int oversampling = 9;

  complex_float *in;

  vector_const_void_star inVec;
  vector_void_star outVec;
  vector_int numIn;

  vector_float taps;
  readFloatSamples(TAP_FILE, taps);
  //for (int i = 0; i < 999; ++i) {
    //taps.push_back(1);
  //}

  int tapsPerFilter = round(((double)taps.size())/numFilters+0.5);
  int totalInSize = numSamples + numFilters*tapsPerFilter;
  for (int i = 0; i < numFilters; ++i) {
    numIn.push_back(numSamples/numFilters+tapsPerFilter);
  }


  mallocOutBufferVector(outVec, numFilters, numSamples/numFilters*oversampling, alignment);

  in = (complex_float*) arch_malloc(12960*sizeof(complex_float), alignment);
  memset(in, 0 , 12960*sizeof(complex_float));

  // Initialize random seed
  srand(time(NULL));

  for (int i = 0; i < 12960; ++i) {
    float re = rand()%100 - 50;
    float im = rand()%100 - 50;
    in[i] = complex_float(re,im);
  }

  //for (unsigned int i = 0; i < totalInSize; ++i) {
    //in[i] = complex_float(i,0);
  //}
  writeComplexSamplesToFile(in, 12960, "inSamples.dat");
  condition_input(in, 45, 288);
  writeComplexSamplesToFile(in, 12960, "inSamplesCond.dat");





  //complexWave(in, totalInSize, 250000);

  for(unsigned int i = 0; i < numFilters; i++)
  {
    inVec.push_back(&in[i*288]);
  }



  gr::filter::pfb_channelizer_ccf::sptr chan = gr::filter::pfb_channelizer_ccf::make(
                                                numFilters, taps, oversampling);
  // Setuo Benchmarking
  struct benchmarkData bmData;
  std::chrono::nanoseconds ns;
  std::chrono::high_resolution_clock::time_point start;
  //chan->print_taps();
  int producedSamples = 0;
  for(unsigned int i = 0; i < numRuns; ++i)
  {

    using namespace std::chrono;

    start = high_resolution_clock::now();
    producedSamples = chan->general_work(numSamples/numFilters*oversampling, numIn,
                       inVec, outVec);
    ns = duration_cast<nanoseconds>(high_resolution_clock::now()-start);

    std::cout << "Produced Samples: " << producedSamples << "\n";

    bmData.average += ((float)ns.count())/numRuns;
    if(ns.count() < bmData.min) bmData.min = ns.count();
    if(ns.count() > bmData.max) bmData.max = ns.count();

  }
  writeComplexSamplesToFile((complex_float*) outVec[0], producedSamples, "outSamples0.dat");
  writeComplexSamplesToFile((complex_float*) outVec[1], producedSamples, "outSamples1.dat");
  writeComplexSamplesToFile((complex_float*) outVec[2], producedSamples, "outSamples2.dat");
  writeComplexSamplesToFile((complex_float*) outVec[5], producedSamples, "outSamples5.dat");
  writeComplexSamplesToFile((complex_float*) outVec[7], producedSamples, "outSamples7.dat");
  writeComplexSamplesToFile((complex_float*) outVec[10], producedSamples, "outSamples10.dat");
  writeComplexSamplesToFile((complex_float*) outVec[30], producedSamples, "outSamples30.dat");


  writeBenchmark(bmData.average, bmData.min, bmData.max, numSamples);

  free(in);
  freeOutBufferVector(outVec);
  return 0;
}
