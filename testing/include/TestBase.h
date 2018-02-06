#ifndef TEST_BASE_H_INCLUDED
#define TEST_BASE_H_INCLUDED

#include <string>
#include <iostream>
#include "TestResult.h"


/****************************************************************************//*!
* @class TestBase
* @brief This is the base class for all tests that are performed
* @details A pure virtual class that is used as a baseclass for all tests. This allows for a generic approach whehn generating and executing tests.
*/
namespace dlr_test_suite
{

  class TestBase
  {
    public:

      virtual ~TestBase(){};

      /*****************************************************************************//*!
      * @brief Virtual run function
      * @brief this class should be implemented by the child classes and used to run the test(s) inside the child class
      * @param in TestResult *result pointer to a Testresult object that can store and display test results
      ***************************************************************************/
      virtual void run(TestResult *result) = 0;

      /*****************************************************************************//*!
      * @brief Virtual function to retreive the name of the test
      * @param out string A string that represents the name of the test
      ***************************************************************************/
      virtual std::string getName() = 0;

      /*****************************************************************************//*!
      * @brief Virtual function to get the number of tests stored in the child class
      * @param in int The number of tests inside the child class
      ***************************************************************************/
      virtual int getTestCount() = 0;

  };

}


#endif /* TEST_BASE_H_INCLUDED */
