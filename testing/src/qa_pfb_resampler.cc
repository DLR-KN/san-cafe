#include <cafe_cuda_helper.h>
#include <pfb_arb_resampler.h>
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
  std::vector<std::complex<float>> result_vec;
  std::vector<float> taps;

  // Read the prepared test files
  readBinarySamples<std::complex<float>>(i_file, i_samples);
  readBinarySamples<std::complex<float>>(o_file, o_samples);
  std::string fname("in.txt");
  writeComplexSamplesToFile(&i_samples[0], i_samples.size(), fname);

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
  resampler->set_filter_block_config(i_samples.size() / num_channels);

  // Compute sizes for the buffers on the GPU
  int global_mem_size_i = i_samples.size();
  int global_mem_size_o = 8192 * num_channels;
  result_vec.resize(global_mem_size_o);
  std::cout << "Output Sample Buffer size: " << o_samples.size() << std::endl;
  std::cout << "Global Output Buffer Size: " << global_mem_size_o << std::endl;

  // Allocate the buffers on the GPU
  cafe::cuda_unique_ptr buffer_gpu_i =
      cafe::create_cuda_unique_ptr(global_mem_size_i * sizeof(float2));
  cafe::cuda_unique_ptr buffer_gpu_o =
      cafe::create_cuda_unique_ptr(global_mem_size_o * sizeof(float2));

  cudaError err =
      cudaMemcpy(buffer_gpu_i.get(), &i_samples[0],
                 global_mem_size_i * sizeof(float2), cudaMemcpyHostToDevice);

  if (err != cudaSuccess) {
    std::cout << "Error copying input to GPU\n";
  }

  resampler->filter(i_samples.size() / num_channels, buffer_gpu_i.get(),
                    buffer_gpu_o.get());

  err = cudaMemcpy(&result_vec[0], buffer_gpu_o.get(),
                   result_vec.size() * sizeof(float2), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) {
    std::cout << "Error copying input to GPU\n";
  }

  // Check Samples
  for (unsigned int c = 0; c < num_channels; ++c) {
    std::complex<float> *channel_ptr = &result_vec[c * 8192];

    for (unsigned int i = 0; i < o_samples.size(); ++i) {
      float diff = std::abs(channel_ptr[i] - o_samples[i]);
      if (diff > 1e-3) {
        std::cout << diff << " Diff at " << i << " for channel " << c
                  << std::endl;
        std::cout << "Check not passed\n";
        return -1;
      }
    }
  }

  std::cout << "All checks passed\n";

  return result;
}

} /* namespace pfb_resampler */
