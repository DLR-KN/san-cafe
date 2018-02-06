/*
 * arch_detection.h 
 * Detects the hardware capabillities of the CPU
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
#ifndef _ARCH_H_
#define _ARCH_H_

#include "arch_detection.h"


void *arch_malloc(uint64_t size, size_t alignment );
inline void arch_init()
{
  setup_arch();
}
#endif /* _ARCH_H_  */
