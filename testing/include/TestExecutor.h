#ifndef TEST_EXECUTOR_H_INCLUDED
#define TEST_EXECUTOR_H_INCLUDED

#include <string.h>
#include <stdlib.h>

#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <memory>

#include "TestObj.h"

namespace dlr_test_suite
{

  template<class type>
  class TestExecutor : public TestObj
  {

    typedef void (type::*TestMethod) (); //! The class type of the Method that is run

    private:

      std::string name_;  //! The name of the TestExecutor Object
      TestMethod method_; //! The method that is run

    public:

      TestExecutor(std::string name, TestMethod method) : name_(name), method_(method){};
      ~TestExecutor() {};


      /*****************************************************************************//*!
      * @brief Run Test function
      * @brief Function that runs the actual test method
      * @param in TestResult *result pointer to a Testresult object that can store and display test results
      ***************************************************************************/
      void runTest()
      {
        type *ty = new type();
        (ty->*method_)();
        delete(ty);
      };

      /*****************************************************************************//*!
      * @brief Function to retreive the name of the test
      * @param out string A string that represents the name of the test
      ***************************************************************************/
      std::string getName()
      {
        return name_;
      };

  };

}

#endif /*TEST_RESULTn_H_INCLUDED*/
