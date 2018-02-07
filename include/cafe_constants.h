/******************************************************************************
* cafe_constants.h
* Header file for variable definitions and constants
*
* Copyright {C} {2017}, {Jan Kraemer, DLR}
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
#ifndef CAFE_CONSTANTS_H
#define CAFE_CONSTANTS_H
#include <cuda_runtime.h>

/*! \enum pacs_error_t
 *  @brief Defines return codes for the CUDA specific funcions
 */
typedef enum {

  SUCCESS,             //! Function exited normally
  CUDA_INIT_ERR,       //! Could not initialize the CUDA Device
  HOST_MEM_ALLOC_ERR,  //! Could not allocate memory on the host
  DEV_MEM_ALLOC_ERR,   //! Could not allocate memory on the device
  DEV_COPY_ERR         //! Could not copy data from/to device

} pacs_error_t;

//
/*!@struct pfb_cuda_config
 * @brief Struct that contains the configuration of the PFB CUDA Kernels
 */
struct pfb_cuda_config {
  int fb_blockdim_x;
  int fb_blockdim_y;
  dim3 fb_griddim;
  int shuffle_blockdim_x;
  int shuffle_blockdim_y;
  int shuffle_griddim;
};

#endif /* CAFE_CONSTANTS_H */
