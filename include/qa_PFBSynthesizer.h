#ifndef _QA_PFB_SYNTHESIZER_H
#define _QA_PFB_SYNTHESIZER_H

#include "TestHelperMacros.h"
#include "AssertMacros.h"

class TestObj;

class qa_PFBSynthesizer
{

  START_TEST_SUITE(qa_PFBSynthesizer);
  ADD_TEST(t1,t1);
  END_TEST_SUITE();

private:
  void t1();
};


#endif /* _QA_PFB_SYNTHESIZER_H */
