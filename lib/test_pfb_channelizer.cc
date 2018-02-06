#include<qa_PFBChannelizer.h>
#include<qa_PFBSynthesizer.h>
#include<qa_pfb_arb_resampler.h>
#include <TestRunner.h>
#include <archs.h>
#include <arch_types.h>

#include<iostream>

int main(int argc, char *argv[])
{

  setup_arch();
  dlr_test_suite::TestRunner runner;
  //ADD_SUITE_TO_RUNNER(runner, qa_PFBChannelizer, PFBChannelizer);
  //ADD_SUITE_TO_RUNNER(runner, qa_PFBSynthesizer, PFBSynthesizer);
  ADD_SUITE_TO_RUNNER(runner, qa_pfb_arb_resampler, pfb_arb_resampler);
  runner.run();
  runner.showStatistic(1);

  return 0;
}
