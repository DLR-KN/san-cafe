#ifndef _FILTERBANK_EXECUTE_H
#define _FILTERBANK_EXECUTE_H
#include "cafe_constants.h"

/*****************************************************************************/ /*!
 * @brief Copies the filtertaps to the Device
 * @param in float  *h_taps Pointer to the tap vector in Constant Memory
 * @param in int    prot_filter_size Width of the prototipe filter
 ***************************************************************************/
int copy_filter_taps(float *h_taps, int prot_filter_size);

/*****************************************************************************/ /*!
 * @brief Sets all constant variables that are local on the Device
 * @param in int    *_offset Length of the channel buffer
 * @param in int    *h_shift_width Width of the snake shift of the input
 * @param in float  *h_taps Pointer to the tap vector in Constant Memory
 * @param in int    *taps_per_channel Number of taps per filter in the
 *                                    filterbank
 * @param in int    prot_filter_size Width of the prototipe filter
 ***************************************************************************/
int set_filterbank_constants(int *h_offset, int *h_shift_width, float *h_taps,
                             int *taps_per_channel, int prot_filter_size);

/*****************************************************************************/ /*!
 * @brief Fires the CUDA kernel for the oversampling case
 * @param in float2  *in Pointer to the input buffer on the Device
 * @param in float2  *in Pointer to the output buffer on the Device
 ***************************************************************************/
__global__ void filterbank_execute(float2 *in, float2 *out);

/*****************************************************************************/ /*!
 * @brief Copies the filtertaps to the Device
 * @param in float  *h_taps Pointer to the tap vector in Constant Memory
 * @param in int    prot_filter_size Width of the prototipe filter
 ***************************************************************************/
__global__ void filterbank_no_os_execute(float2 *in, float2 *out);

#endif /* ifndef _FILTERBANK_EXECUTE_H */
