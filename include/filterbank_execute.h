#ifndef _FILTERBANK_EXECUTE_H
#define _FILTERBANK_EXECUTE_H
#include "cafe_constants.h"

/*! @fn set_filterbank_constants
 * @brief Wrapper function to copy the constant symbols
 * @param
 */

int copy_filter_taps(float *h_taps, int prot_filter_size);
int set_filterbank_constants(int *h_offset, int *h_shift_width, float *h_taps,
                             int *taps_per_channel, int prot_filter_size);
__global__ void filterbank_execute(float2 *in, float2 *out);
__global__ void filterbank_no_os_execute(float2 *in, float2 *out);

#endif /* ifndef _FILTERBANK_EXECUTE_H */
