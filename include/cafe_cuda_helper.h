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
#ifndef _CAFE_CUDA_HELPER_H
#define _CAFE_CUDA_HELPER_H
#include <cuda_runtime.h>
#include <memory>

namespace cafe {

/*!
 * @brief Function that initializes the CUDA device. Returns PACS_CUDA_INIT_ERR
 * if it fails.
 */
int cafe_init_cuda();

/*!
 * @brief Callback function to allocate memory on the CUDA device
 * @param[in] size Size of memory to be allocated in bytes
 */
void malloc_cuda(size_t size);

/*!
 * @brief Callback function to free memory on the CUDA device
 * @param[in] ptr Pointer to memory on the CUDA device
 */
void free_cuda(void *ptr);

/*
 * @brief std::unique_ptr typedef with custom allocator/deallocator for
 * allocating memory on a CUDA device
 */
typedef std::unique_ptr<float2, void (*)(void *)> cuda_unique_ptr;

/*!
 * @brief Factory function to create a unique ptr for memory on the CUDA device
 * @param[in] size Size of memory to be allocated in bytes
 */
cuda_unique_ptr create_cuda_unique_ptr(size_t size);
} /* namespace cuda */

#endif /* ifndef _CAFE_CUDA_HELPER_H */
