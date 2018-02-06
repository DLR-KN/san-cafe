#ifndef ARCH_INCLUDE_ARCHTYPES_H_
#define ARCH_INCLUDE_ARCHTYPES_H_

/*

 * Structs and defines for different CPU architectures
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

/*
 * Defines for the bits that are set in the output register of
 * the CPUID instruction when querying processor info
 */
#include <string>

#define MMX    23
#define SSE    25
#define SSE2   26
#define SSE3   0
#define SSE4_1 19
#define SSE4_2 20
#define AVX    28 

// Bit set if the OS supports the xgetbv instruction
#define OS_XGETBV 27

/*
 * Struct to store information of the cpu type
 */
struct cpu_type {

  std::string machine;
  size_t alignment;
  uint32_t arch_support;

};

extern struct cpu_type current_machine;

/*
 * ENUM for the information type to be read from the CPUID instruction
 */
typedef enum{
GET_VENDOR_ID,
GET_PROC_INFO
} CPUID_INFO_TYPE;

/*
 * ENUM for the ASM register type (eax,ebx,ecx,edx)
 */
typedef enum{
EAX,
EBX,
ECX,
EDX
} ASM_REGISTER_TYPE;

inline size_t get_alignment()
{
  return current_machine.alignment;
};

inline uint32_t get_arch_support()
{
  return current_machine.arch_support;
}

#endif
