#ifndef READ_SAMPLES_H
#define READ_SAMPLES_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>


void getFloat(std::ifstream &file, std::vector<float> &samples);

int readFloatSamples(const std::string &fileName, std::vector<float> &samples);

/******************************************************************************
 * The Template section
 * Here one can find all definitions of function templates
 *****************************************************************************/
template<typename T>
void getComplex(std::ifstream &file, std::vector<std::complex<T>> &samples)
{
  std::string line;
  // Temp buffer for the complex number
  // Pointer to the first char after we read the real part
  size_t idx = 0;
  while(getline(file, line))
  {
    T real = static_cast<T> (std::stof(line, &idx));
    T im   = static_cast<T> (std::stof(line.substr(idx+1), &idx));
    std::complex<T> cSample(real, im);
    samples.push_back(cSample);
  }

}

template<typename T>
int
readComplexSamples(const std::string &fileName, std::vector<std::complex<T>> &samples)
{

  std::cout << "Opening File " << fileName << "\n";
  std::ifstream sampleFile(fileName);
  if(!sampleFile.is_open())
  {
    std::cout << "Error: unable to open file!" << "\n";
    return -1;
  }

  getComplex(sampleFile, samples);
  return 0;

}

template<typename T>
int readBinarySamples(const std::string &fileName, std::vector<T> &samples)
{
  std::cout << "Opening file " << fileName << "\n";
  std::ifstream sampleFile(fileName, std::ios_base::in | std::ios_base::binary);

  if(!sampleFile.is_open()) {
    std::cout << "Error: Unable to open file " << fileName << "\n";
    return -1;
  }

  //get length of file
  sampleFile.seekg(0, sampleFile.end);
  int length = sampleFile.tellg();
  sampleFile.seekg(0, sampleFile.beg);
  int num_samples = length/sizeof(T);

  samples.resize(num_samples, 0);

  sampleFile.read((char*)samples.data(), length);

  if(sampleFile)
    std::cout << "Success:: Samples read correctly\n";
  else
    std::cout << "Error: Could not read the samples\n";

  sampleFile.close();

  return 0;

}

template<typename T>
int readBinaryComplexSamples(const std::string filename_real, const std::string filename_imag,
                             std::vector<std::complex<T>> &samples)
{

  std::ifstream sampleFileReal(filename_real, std::ios_base::in | std::ios_base::binary);
  std::ifstream sampleFileImag(filename_real, std::ios_base::in | std::ios_base::binary);

  if(!sampleFileReal.is_open() && !sampleFileImag.is_open()) {
    std::cout << "Unable to open files " << filename_real << " & " << filename_imag << std::endl;
    return -1;
  }

  std::vector<T> realBuffer;
  std::vector<T> imagBuffer;

  // Get Length for real File
  sampleFileReal.seekg(0, sampleFileReal.end);
  int lengthReal = sampleFileReal.tellg();
  sampleFileReal.seekg(0, sampleFileReal.beg);

  // Get Length for imag File
  sampleFileImag.seekg(0, sampleFileImag.end);
  int lengthImag = sampleFileImag.tellg();
  sampleFileImag.seekg(0, sampleFileImag.beg);

  if(lengthImag != lengthReal) {
    std::cerr <<"Files do not have identical length!\n";
    return -1;
  }

  int numSamples = lengthReal/sizeof(T);

  realBuffer.resize(numSamples, 0);
  imagBuffer.resize(numSamples, 0);

  sampleFileReal.read((char*)realBuffer.data(), lengthReal);
  if(!sampleFileReal) {
    std::cerr << "Error reading Real samples\n";
    return -1;
  }
  sampleFileImag.read((char*)imagBuffer.data(), lengthReal);

  if(!sampleFileImag) {
    std::cerr << "Error reading Complex samples\n";
    return -1;
  }

  sampleFileReal.close();
  sampleFileImag.close();


  samples.resize(numSamples, 0);
  for (int i = 0; i < numSamples; ++i) {
    samples[i] = std::complex<T>(realBuffer[i], imagBuffer[i]);
  }

  std::cout << "Samples read correctly\n";

  return numSamples;

}

#endif /* READ_SAMPLES_H */
