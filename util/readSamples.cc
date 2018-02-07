#include "readSamples.h"

/*****************************************************************************/
void
getFloat(std::ifstream &file, std::vector<float> &samples)
{
  std::string line;
  while(getline(file, line))
  {
    samples.push_back(std::stof(line, nullptr));
  }
}

/*****************************************************************************/
int
readFloatSamples(const std::string &fileName, std::vector<float> &samples)
{
  std::cout << "Opening File " << fileName << "\n";
  std::ifstream sampleFile(fileName);

  if(!sampleFile.is_open())
  {
    std::cout << "Unable to open file!" << "\n";
    return -1;
  }

  getFloat(sampleFile, samples);

  return 0;

}

/*****************************************************************************/
