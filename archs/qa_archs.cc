#include "qa_archs.h"
#include "archs.h"
#include <iostream>
#include "string.h"
#include "AssertMacros.h"

// Check the right arch support for PACS
void qa_archs::t2()
{
  unsigned int reg[4];
  
  unsigned int expected_arch_support = 255;

  memset(reg, 0 , sizeof(unsigned int)*4);

  setup_arch();
  ASSERT_INT_EQUAL(expected_arch_support, get_arch_support());
}

// Check the right alignment for PACS
void qa_archs::t3()
{
  unsigned int reg[4];

  memset(reg, 0 , sizeof(unsigned int)*4);

  setup_arch();
  ASSERT_INT_EQUAL(expected_alignment, get_alignment());
}

// Check the space uni system for PACS
void qa_archs::t4()
{

  space_uni_check();

}
