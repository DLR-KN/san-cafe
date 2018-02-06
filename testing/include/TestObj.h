#ifndef TEST_OBJ_H_INCLUDED
#define TEST_OBJ_H_INCLUDED

#include <string.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <memory>

#include "TestException.h"
#include "TestResult.h"
#include "TestBase.h"

namespace dlr_test_suite
{

  class TestObj : public TestBase{

    private:

      std::string name_;

    public:

      TestObj(){};
      TestObj(std::string name) : name_(name){};
      virtual ~TestObj() {};

      /*****************************************************************************//*!
      * @brief Function that runs the appropriate test
      * @details Starts the testRun and and handles success/failure/error management
      * @param in TestResult *result pointer to a Testresult object that can store and display test results
      ***************************************************************************/
      void run(TestResult *result)
      {
         result->startTestRun();
        try
        {

          runTest();
          result->addSuccess(this);
          result->reportSuccess();

        }
        catch(TestException &e)
        {
          //std::cout << "TestException " << e.what() << std::endl;
          result->addFailure(this);
          result->reportFailure();
        }
        catch(...)
        {
          result->addError(this);
          result->reportError();
        }

        result->endTestRun();

      };

      /*****************************************************************************//*!
      * @brief Returns the name of the TestObj
      * @param out std::string name_ Name of the TestObj
      ***************************************************************************/
      virtual std::string getName()
      {
        return name_;
      };

      /*****************************************************************************//*!
      * @brief Returns the Number of tests in TestObj
      * @details As as TestObj has only one test, this count always returns 1!
      * @param out int 1
      ***************************************************************************/
      virtual int getTestCount()
      {
        return 1;
      };

      /*****************************************************************************//*!
      * @brief Virtual run Test function
      * @brief Function that runs the actual testmethod
      * @param in TestResult *result pointer to a Testresult object that can store and display test results
      ***************************************************************************/
      virtual void runTest() =0;

  };

}

#endif /*TEST_OBJ_H_INCLUDED*/
