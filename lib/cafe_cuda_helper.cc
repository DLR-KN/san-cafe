/******************************************************************************
* test_helper_functions.h
* Header file for the helper functions
*
* Copyright {C} {2016}, {Jan Kraemer, DLR}
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser  General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
******************************************************************************/
#include <cafe_constants.h>
#include <cafe_cuda_helper.h>
#include <iomanip>
#include <iostream>
#include <vector>

namespace cafe
{
int cafe_init_cuda()
{
  std::cout << "\nGetting CUDA Device count...";
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess) {
    std::cout << "failed\n";
    return CUDA_INIT_ERR;
  }

  if (device_count == 0) {
    std::cout << "none\n";
    return CUDA_INIT_ERR;
  } else {
    std::cout << device_count << "\n";
  }

  // We only want the device 1
  const int device = 0;
  cudaSetDevice(0);

  // Get the device properties
  cudaDeviceProp device_properties;
  cudaGetDeviceProperties(&device_properties, device);

  std::cout << "\nDevice: " << device_properties.name << "\n";
  std::cout << "CUDA capabilities : " << device_properties.major << "."
            << device_properties.minor << "\n";

  return SUCCESS;
}

void malloc_cuda(void **ptr, size_t size) { cudaMalloc(ptr, size); }

void free_cuda(void *ptr)
{
#if DEBUG_PRINT
//      std::cout << " Erasing memory at address : " << std::hex << (long long)
//      ptr << "\n";
#endif
  cudaFree(ptr);
}

cuda_unique_ptr create_cuda_unique_ptr(size_t size)
{
  // Is this the real life, or just evil hackery :)
  float2 *tmp_ptr;
  cudaError err = cudaMalloc((void **)&tmp_ptr, size);
  if (err != cudaSuccess) {
    std::cout << "cudaMalloc failed with " << cudaGetErrorString(err)
              << std::endl;
    return cuda_unique_ptr(nullptr, free_cuda);
  }

  return cuda_unique_ptr(tmp_ptr, free_cuda);
}

} /* namespace cuda */
