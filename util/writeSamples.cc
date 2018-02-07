#include "writeSamples.h"

// Default implementation
void writeComplexSamplesToFile(std::complex<float> *data, unsigned int len,
                               const std::string &filename)
{
  std::ofstream file(filename);

  if (file.is_open()) {
    for (unsigned int i = 0; i < len; ++i) {
      file << data[i].real();
      if (data[i].imag() < 0) {
        file << data[i].imag() << "j\n";
      } else {
        file << "+" << data[i].imag() << "j\n";
      }
    }
  } else {
    std::cout << "Unable to open file\n";
  }
}

/*****************************************************************************/
void writeComplexSamplesToFile(std::complex<float> *data, unsigned int len,
                               const std::string &filename, writeMode mode)
{
  std::ofstream file(filename, modeVec[mode]);

  if (file.is_open()) {
    for (unsigned int i = 0; i < len; ++i) {
      file << data[i].real();
      if (data[i].imag() < 0) {
        file << data[i].imag() << "j\n";
      } else {
        file << "+" << data[i].imag() << "j\n";
      }
    }
  } else {
    std::cout << "Unable to open file\n";
  }
}
