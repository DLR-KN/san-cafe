#include "arch_detection.h"
#include "arch_types.h"
#include <iostream>
#include "string.h"

#define SU

int main()
{
  unsigned int reg[4];

  memset(reg, 0 , sizeof(unsigned int)*4);

  setup_arch();
  std::cout << "Current architecture support: " << get_arch_support() << std::endl;
  std::cout << "Current alignment: "  << get_alignment() << std::endl;

#ifdef SU
  space_uni_check();
#endif
  return 0;
}
