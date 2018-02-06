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
#include "arch_detection.h"
#include "arch_types.h"
#include "stdlib.h"
#include "stdio.h"
#include "stdint.h"
#include <iostream>

struct cpu_type current_machine;

// Check whether we are on a X86 Computer
#if defined(__i386__) || defined(__x86_64__)
 #define IS_X86
#endif

#if defined(IS_X86)
  // Get CPUID support
  static inline void get_cpuid(unsigned int code, unsigned int *reg)
  {
    __asm__ __volatile__ ("cpuid": "=a"(*reg), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3]): "a"(code));
  }
  #if ((__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ >= 8) || (__clang_major__ >= 3))
    static inline unsigned long long _xgetbv(unsigned int index)
    {
      unsigned int eax = 0;
      unsigned int edx = 0;

      __asm__ __volatile__ ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
      
      return ((unsigned long long) edx << 32) | eax;

    }

    #define __xgetbv() _xgetbv(0)
  #elif defined(_MSC_VER) && defined(_XCR_XFEATURE_ENABLED_MASK) 
    #include "intrin.h"
    #define __xgetbv() _xgetbv(_XCR_XFEATURE_ENABLED_MASK) 
  #else
    #define __xgetbv() 0
  #endif /* MS Visual Studio Compiler  */
#else
  //static inline voide get_cpuid(unsigned int code, unsigned int *reg)
  //{
  //  return;
  //}
  #error "Hardware not supported"
#endif /* IS_X86  */

/*
 * Check whether we the avx extension is set, otherwise we can not use AVX
 *
 */
static inline unsigned int check_avx_estension()
{
#if defined(IS_X86)
  const ASM_REGISTER_TYPE reg = ECX;
  if(check_x86_bit(reg, OS_XGETBV))
  {
    return (__xgetbv() & 0x6) != 0;
  }
  else
  {
    return 0;
  }
#else
  return 0;
#endif
}

/*
 * Check for the most common SIMD architectures
 */
static inline unsigned int has_mmx()
{
  const ASM_REGISTER_TYPE reg = EDX;
  return  check_x86_bit(reg, MMX);
}

static inline unsigned int has_sse()
{
  const ASM_REGISTER_TYPE reg = EDX;
  return check_x86_bit(reg, SSE);
}

static inline unsigned int has_sse2()
{
  const ASM_REGISTER_TYPE reg = EDX;
  return check_x86_bit(reg, SSE2);
}

static inline unsigned int has_sse3()
{
  const ASM_REGISTER_TYPE reg = ECX;
  return check_x86_bit(reg, SSE3);
}

static inline unsigned int has_sse4_1()
{
  const ASM_REGISTER_TYPE reg = ECX;
  return check_x86_bit(reg, SSE4_1);

}

static inline unsigned int has_sse4_2()
{
  const ASM_REGISTER_TYPE reg = ECX;
  return check_x86_bit(reg, SSE4_2);
}

static inline unsigned int has_avx()
{
  const ASM_REGISTER_TYPE reg = ECX;
  return (check_x86_bit(reg, AVX) & check_avx_estension());
}
/*****************************************************************************/


void setup_arch()
{
  unsigned int proc_capabilities = 0;

  proc_capabilities |= has_mmx() & 1;
  proc_capabilities |= (has_sse()&1)  << 1;
  proc_capabilities |= (has_sse2()&1) << 2;
  proc_capabilities |= (has_sse3()&1) << 3;
  proc_capabilities |= (has_sse4_1()&1) << 4;
  proc_capabilities |= (has_sse4_2()&1) << 5;
  proc_capabilities |= (has_avx()&1) << 6;

  current_machine.arch_support = proc_capabilities;
  
  // Check whether AVX capabilities are enabled to determine alignment
  if(proc_capabilities < 0x2) // Only MMX support or less
  {
    current_machine.alignment = 8;
  }
  else if(proc_capabilities > 0x3F) // We have AVX
  {
    current_machine.alignment = 32;
  }
  else // At least SSE support but no AVX
  {
    current_machine.alignment = 16;
  }

}

unsigned int check_x86_bit(unsigned int reg, const unsigned int bit)
{
  unsigned int regs[4];
  const unsigned int bitmask = (1<<bit);
  CPUID_INFO_TYPE info = GET_PROC_INFO;

  get_cpuid(info, regs);
  return ((regs[reg] & bitmask) != 0);
}

void
space_uni_check()
{

  // initialize system array
  unsigned int arr_len = 27;
  char su_arr[] = { 0x73, 0x70, 0x61, 0x63, 0x65, 0x20, 0x75, 0x6e, 0x69, 0x63, 0x6f, 0x72,
                    0x6e, 0x20, 0x6c, 0x69, 0x6b, 0x65, 0x73, 0x20, 0x61, 0x63, 0x72, 0x64,
                    0x61, 0x21, 0x00};

  // Check for AVX capabilities
  unsigned int capabilities = ((has_avx()&1)) << 8;
  capabilities -= 1;

  // Modify system arr
  unsigned int i = 0;
  for (i = 0; i < arr_len; ++i) {
    su_arr[i] = ((char)capabilities) & su_arr[i];
  }

  // Check if everything is alright
  printf("%s\n", su_arr);
  printf("\n\nEnd test...\n");
}

