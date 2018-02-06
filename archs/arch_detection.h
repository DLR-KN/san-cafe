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
#ifndef _ARCH_DETECTION_H_
#define _ARCH_DETECTION_H_

#include "stdlib.h"
#include "stdint.h"
#include "stdio.h"

// Setup the arch structs and values
void setup_arch();
void setup_machine_name();
/*
 * Checks if a bit in the CPUID is set, returns 1 or 0 if successfull, -1 on error
 * @param in reg CPUID returns 4 32 bit registers(eax,ebx,ecx,edx), this specifies the register to read from
 * @param in bit This specifies the bit to be checked in the choses register
 */
unsigned int check_x86_bit(unsigned int reg, const unsigned int bit );

void space_uni_check();

#endif /* _ARCH_DETECTION_H_  */
