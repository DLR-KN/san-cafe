#ifndef TEST_RUNNER_H_INCLUDED
#define TEST_RUNNER_H_INCLUDED

#include <iostream>
#include <stdlib.h>

#include <string>
#include <vector>
#include <exception>

#include "TestResult.h"
#include "TestSuite.h"

class TestBase;

namespace dlr_test_suite
{

  class TestRunner
  {

    public:
      TestRunner()
      {
      };

      ~TestRunner()
      {
      };

      /*****************************************************************************//*!
      * @brief Adds a new test function
      * @param in TestBase *test
      ***************************************************************************/
      void addTest(TestBase *test)
      {
        testList_.push_back(test);
      };

      /*****************************************************************************//*!
      * @brief Runs the TestObjects/TestSuites
      * @details Setups the result for the tests and runs all necessary test methods
      ***************************************************************************/
      void run()
      {

        result_.setUp(getTestCount());

        for(std::vector<TestBase*>::iterator it = testList_.begin(); it != testList_.end(); ++it)
        {
          TestBase *test =  *it;
          test->run( &result_);
        }

      };

      /*****************************************************************************//*!
      * @brief Shows the accumulated statistics
      * @param in int verboseLevel The degree of information that is displayed
      ***************************************************************************/
      void showStatistic(int verboseLevel)
      {
        int performedTests = 0;

        performedTests = result_.testFailures() + result_.testErrors() + result_.testSuccesses();

        std::cout << "Performed " << performedTests << "/" << getTestCount() << " Tests" << std::endl;
        std::cout << result_.testFailures() <<"/"<< performedTests << " Tests failed" << std::endl;

        if(verboseLevel)
        {
          for(auto&& it : result_.getFailureList())
          {
            std::cout << it->getName() << " failed " << std::endl;
          }
        }
      };

      /*****************************************************************************//*!
      * @brief Gets the number of tests in this TestRunner
      * @param out int suiteTestCount
      ***************************************************************************/
      int getTestCount()
      {

        int suiteTestCount = 0;

        for(std::vector<TestBase*>::iterator it = testList_.begin(); it != testList_.end(); ++it)
        {
          TestBase *test =  *it;
          suiteTestCount += test->getTestCount();
        }

        return suiteTestCount;

      };


    private:

      std::vector<TestBase*> testList_; //! Vector containing all the tests/testSuites
      TestResult result_;               //! The results of all the tests
  };

}

#endif /*TEST_RESULTn_H_INCLUDED*/
