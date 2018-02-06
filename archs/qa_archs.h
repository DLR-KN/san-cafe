#ifndef _QA_ARCHS_H_
#define _QA_ARCHS_H_
#include "TestHelperMacros"

class TestObj;

class qa_Archs
{
  START_TEST_SUITE(qa_Archs);
  ADD_TEST(t2,t2);
  ADD_TEST(t3,t3);
  END_TEST_SUITE();

private:
  
  void t2();
  void t3();w

}
#endif /* _QA_ARCHS_H_  */
