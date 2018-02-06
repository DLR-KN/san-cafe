#ifndef TEST_EXCEPTION_H_INCLUDED
#define TEST_EXCEPTION_H_INCLUDED

#include <iostream>
#include <stdlib.h>

#include <string>
#include <vector>
#include <exception>

namespace dlr_test_suite
{

  class TestException : std::exception
  {

    public:

      /*****************************************************************************//*!
      * @brief Constructor for the Exception
      * @param in std::string message A message that explains the reason for the exception
      ***************************************************************************/
    TestException(std::string message) :
    message_(message)
    {
    };

    /*****************************************************************************//*!
    * @brief Destructor for the Exception
    * @details It also throws :D
    ***************************************************************************/
    ~TestException() throw()
    {
    };

    /*****************************************************************************//*!
    * @brief What functions
    * @details returns a const char ptr that contains the exception message
    * @param out const char*
    ***************************************************************************/
    const char* what() const throw()
    {

      return message_.c_str();

    };

    /*****************************************************************************//*!
    * @brief Get the Exception Message functions
    * @details returns a string that contains the exception message
    * @param out std::string
    ***************************************************************************/
    std::string getMessage()
    {
      return message_;
    };

    private:
    std::string message_;  //! A brief explanation of the current Exception

  };
}
#endif /*TEST_Exception_H_INCLUDED*/
