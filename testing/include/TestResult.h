#ifndef TEST_RESULT_H_INCLUDED
#define TEST_RESULT_H_INCLUDED

#include <iostream>
#include <stdlib.h>

#include <string>
#include <vector>
#include <exception>



namespace dlr_test_suite
{

  class TestBase;

  class TestResult
  {

    private:

      std::vector<TestBase*> failureList_;  //! The vector with all the tests that failed the test run
      std::vector<TestBase*> errorList_;    //! The vector with all tests that experienced unexpected errors (Errors that are not check for)
      std::vector<TestBase*> successList_;     //! The vector Tests that succeeded
      int totalTestCount_;                  //! The number of tests in the test run
      int currentTestNumber_;               //! The number of the currently running test

    public:
      TestResult()
      {
      };

      ~TestResult()
      {
      };

      /*****************************************************************************//*!
      * @brief Start a new Test Setup
      * @param in int totalTestCount The total number of tests that are going to be performed
      ***************************************************************************/
      void setUp(int totalTestCount)
      {
        totalTestCount_ = totalTestCount;
        currentTestNumber_ = 0;
      };

      /*****************************************************************************//*!
      * @brief Add a Test to the Success List
      * @param in TestBase * test
      ***************************************************************************/
      void addSuccess(TestBase *test)
      {
        successList_.push_back(test);
      };

      /*****************************************************************************//*!
      * @brief Add a Test to the Failure List
      * @param in TestBase * test
      ***************************************************************************/
      void addFailure(TestBase *test)
      {
        failureList_.push_back(test);
      };

      /*****************************************************************************//*!
      * @brief Add a Test to the Error List
      * @param in TestBase * test
      ***************************************************************************/
      void addError(TestBase *test)
      {
        errorList_.push_back(test);
      };

      /*****************************************************************************//*!
      * @brief What functions
      * @details returns a const char ptr
      * @param out int Number of tests that failed in this test run
      ***************************************************************************/
      int testSuccesses()
      {
        return successList_.size();
      };

      /*****************************************************************************//*!
      * @brief What functions
      * @details returns a const char ptr
      * @param out int Number of tests that failed in this test run
      ***************************************************************************/
      int testFailures()
      {
        return failureList_.size();
      };

      /*****************************************************************************//*!
      * @brief What functions
      * @details returns a const char ptr
      * @param out int Number of tests that failed in this test run
      ***************************************************************************/
      int testErrors()
      {
        return errorList_.size();
      };

      /*****************************************************************************//*!
      * @brief Outputs the progress of the currently running test and test run
      ***************************************************************************/
      void startTestRun()
      {
        outputProgress(++currentTestNumber_);
      };

      /*****************************************************************************//*!
      * @brief Lazy, just lazy
      ***************************************************************************/
      void endTestRun()
      {
        std::cout << std::endl;
      };

      /*****************************************************************************//*!
      * @brief Show the progress of the current running test run
      * @param in int testNumber Number of the currently running test in the test run
      ***************************************************************************/
      void outputProgress(int testNumber)
      {
        std::cout << "Running Test................ " << testNumber << "/" << totalTestCount_ << std::endl;
      };

      /*****************************************************************************//*!
      * @brief Display that a test was performed successfully
      ***************************************************************************/
      void reportSuccess()
      {
        std::cout << std::endl <<"Passed" << std::endl;
      };

      /*****************************************************************************//*!
      * @brief Display that a test failed the assertions or thrown a dlr_test_suite::Exception
      ***************************************************************************/
      void reportFailure()
      {
        std::cout << std::endl << "Failed" << std::endl;
      };

      /*****************************************************************************//*!
      * @brief Display that a test was failed due to an unknown exception
      ***************************************************************************/
      void reportError()
      {
        std::cout << std::endl << "Failed with exceptional error" << std::endl;
      };

      /*****************************************************************************//*!
      * @brief Returns the vector with the failed tests
      * @param out std::vector<TestBase*>
      ***************************************************************************/
      std::vector<TestBase*> getFailureList()
      {
        return failureList_;
      }

  };

}

#endif /*TEST_RESULTn_H_INCLUDED*/
