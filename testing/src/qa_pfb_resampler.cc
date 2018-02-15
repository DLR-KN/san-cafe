#include <pfb_arb_resampler.h>
#include <cafe_cuda_helper.h>
#include <testing/qa_pfb_resampler.h>
#include <util/readSamples.h>
#include <util/writeSamples.h>

namespace pfb_resampler
{
void resize_input_vector(std::vector<std::complex<float>> &vec,
                         unsigned int factor)
{
  const unsigned int old_size = vec.size();
  const unsigned int new_size = old_size * factor;

  vec.resize(new_size);

  for (unsigned int i = old_size; i < new_size; ++i) {
    vec[i] = vec[i % old_size];
  }
}

int test_1()
{
  const std::string name("Basic Test");

  const unsigned int num_channels = 32;
  const unsigned int num_filters = 30;
  const double rate = 15.0 / 16.0;

  int result = 0;

  const std::string i_file = test_dir + "resampler_test_in.bin";
  const std::string o_file = test_dir + "resampler_test_out.bin";
  const std::string tap_file = test_dir + "resampler_taps.txt";

  std::vector<std::complex<float>> i_samples;
  std::vector<std::complex<float>> o_samples;
  std::vector<float> taps;

  // Read the prepared test files
  readBinarySamples<std::complex<float>>(i_file, i_samples);
  readBinarySamples<std::complex<float>>(o_file, o_samples);

  // Read the prepared Filter taps
  readFloatSamples(tap_file, taps);

  // Expand the input sample vector to emulate several channels
  resize_input_vector(i_samples, num_channels);

  // Initialize the GPU
  std::cout << "Initializing the GPU\n";
  cafe::cafe_init_cuda();

  // Inititalize the pfb_arb_resampler
  filter::pfb_arb_resampler::pfb_arb_resampler_sptr resampler =
      filter::pfb_arb_resampler::pfb_arb_resampler_factory(taps, num_filters,
                                                           rate, num_channels);

  return result;
}

} /* namespace pfb_resampler */
