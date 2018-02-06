#ifndef TEST_SUITE_H_INCLUDED
#define TEST_SUITE_H_INCLUDED

#include <iostream>
#include <stdlib.h>

#include <string>
#include <vector>
#include <memory>

#include "TestBase.h"

class TestBase;

namespace dlr_test_suite
{

  class TestSuite : public TestBase{

    private:

      std::string suiteName_;
      std::vector<std::shared_ptr<TestBase>> testList_;
      int testCount_;


    public:
      TestSuite(std::string suiteName ) : suiteName_(suiteName), testCount_(0) {};
      ~TestSuite()
      {

      };            // Inline because not much to do

      /*****************************************************************************//*!
      * @brief Function that runs the tests inside the suite
      * @details Calls the run method implemented for each test in the suite
      * @param in TestResult *result pointer to a Testresult object that can store and display test results
      ***************************************************************************/
      void run(TestResult *result)
      {

        for(std::vector<std::shared_ptr<TestBase>>::iterator it = testList_.begin(); it != testList_.end(); ++it)
        {
          TestBase *test =  it->get();
          test->run(result);
        }
      };

      /*****************************************************************************//*!
      * @brief Returns the name of the TestSuite
      * @param out std::string name_ Name of the TestSuite
      ***************************************************************************/
      std::string getName()
      {
        return suiteName_;
      };

      /*****************************************************************************//*!
      * @brief Returns the Number of tests in the TestSuite
      * @param out int suiteTestCount Number of tests in the suite
      ***************************************************************************/
      int getTestCount()
      {

        int suiteTestCount = 0;

        for(std::vector<std::shared_ptr<TestBase>>::iterator it = testList_.begin(); it != testList_.end(); ++it)
        {
          TestBase *test =  it->get();
          suiteTestCount += test->getTestCount();
        }

        return suiteTestCount;

      };

      /*****************************************************************************//*!
      * @brief Adds a test to the TestSuite
      * @param TestBase *test The test added to the suite
      **************************************************************************/
      void addTest(std::shared_ptr<TestBase> test)
      {
        testList_.push_back(test);
      };

  };

}

#endif /*TEST_SUITE_H_INCLUDED*/
