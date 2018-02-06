/*
 * archs.cc 
 * Functions that are arch specific, relies on arch detection to automatically
 * detect archs specification
 *
 * Copyright {Jan Kraemer, DLR}
 *
 * This Source Code Form is subject to the
 * terms of the Mozilla Public License, v.
 * 2.0. If a copy of the MPL was not
 * distributed with this file, You can
 * obtain one at
 * http://mozilla.org/MPL/2.0/.
 */
#include "archs.h"
#include <iostream>

void *arch_malloc(size_t size, size_t alignment )
{
  
  int error;
  void *ptr;

  error = posix_memalign(&ptr, alignment, size);

  if(error == EINVAL)
  {
    std::cout << "arch_malloc error: Alignment should be a power of two and multiple of sizeof(void*)" << std::endl;
    return NULL;
  }
  else if(error == ENOMEM)
  {
    std::cout << "arch_malloc error: Insufficient memory for allocation request" << std::endl;
    return NULL;
  }

  return ptr;

}
