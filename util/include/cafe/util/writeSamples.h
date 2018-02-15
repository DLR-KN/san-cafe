#ifndef _WRITE_SAMPLES_TO_FILE_H_
#define _WRITE_SAMPLES_TO_FILE_H_
#include <complex>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>

enum writeMode { OVERWRITE, APPEND };

const std::ios_base::openmode modeVec[] = {std::ios_base::out,
                                           std::ios_base::app};

// Default implementation
void writeComplexSamplesToFile(std::complex<float> *data, unsigned int len,
                               const std::string &filename);
/*****************************************************************************/
void writeComplexSamplesToFile(std::complex<float> *data, unsigned int len,
                               const std::string &filename, writeMode mode);
/*****************************************************************************/
template <typename T>
int writeSamplesToFile(const T *data, unsigned int len,
                       const std::string &filename, const std::string delimiter,
                       writeMode mode)
{
  int ret = 0;
  std::ofstream file(filename, modeVec[mode]);

  if (file.is_open()) {
    for (unsigned int i = 0; i < len; ++i) {
      file << data[i] << delimiter;
    }

  } else {
    std::cout << "Unable to open file\n";

    ret = 1;
  }

  return ret;
}

// Default implementation
template <typename T>
static int writeSamplesToFile(const T *data, unsigned int len,
                              const std::string &filename)
{
  int ret = 0;

  // Default behavior is to overwrite
  std::ofstream file(filename, std::ofstream::out);
  char default_delim = static_cast<char>(0);

  if (file.is_open()) {
    for (unsigned int i = 0; i < len; ++i) {
      file << data[i] << default_delim;
    }

  } else {
    std::cout << "Unable to open file\n";
    ret = 1;
  }

  return ret;
}

/*****************************************************************************/
template <typename T>
int writeBinarySamples(T *data, unsigned int len, const std::string &filename,
                       writeMode mode)
{
  int ret = 0;

  std::ofstream sampleFile(filename, modeVec[mode]);

  if (!sampleFile.is_open()) {
    std::cout << "Error: Unable to opne file " << filename << "\n";
    ret = 1;
    return ret;
  }

  unsigned int file_size = len * sizeof(T);

  sampleFile.write((char *)data, file_size);

  sampleFile.close();

  return ret;
}

/*****************************************************************************/
template <typename T>
int writeBinarySamples(std::vector<T> data, const std::string &filename)
{
  int ret = 0;
  std::ofstream sampleFile(filename,
                           std::ios_base::out | std::ios_base::binary);

  if (!sampleFile.is_open()) {
    std::cout << "Error: Unable to opne file " << filename << "\n";
    ret = 1;
    return ret;
  }

  unsigned int file_size = data.size() * sizeof(T);

  sampleFile.write((char *)data.data(), file_size);

  sampleFile.close();

  return ret;
}
#endif
