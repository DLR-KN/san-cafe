/*! \file TestHelperMacros.h
    \brief Macros to test help set up a TestSuite

*/
#ifndef TEST_HELPER_MACROS_INCLUDED
#define TEST_HELPER_MACROS_INCLUDED

#include "TestSuite.h"
#include "TestExecutor.h"
#include <memory>
class TestSuite;

/*!
* Starts a test Suite
*/
#define START_TEST_SUITE(SuiteName)                                                                 \
    typedef SuiteName SuiteClass;                                                                   \
    public:                                                                                         \
                                                                                                    \
    static dlr_test_suite::TestSuite *suite()                                                       \
    {                                                                                               \
      std::unique_ptr<dlr_test_suite::TestSuite> suite(new dlr_test_suite::TestSuite(#SuiteName));  \
/*!
* Adds a test name testName to the TestSuite
*/
#define ADD_TEST(testName, variable)                                                                          \
  std::shared_ptr<dlr_test_suite::TestExecutor<SuiteClass>> variable(new dlr_test_suite::TestExecutor<SuiteClass>(std::string(#testName), &SuiteClass::testName)); \
  suite->addTest(variable);   \

/*!
* Finish the setup for the testSuite by releasing the unique ptr
*/
#define END_TEST_SUITE()                                                                            \
  return suite.release();                                                                           \
};                                                                                                  \


#define ADD_SUITE_TO_RUNNER(runner,suiteClassName, variable)                                                      \
  std::unique_ptr<dlr_test_suite::TestSuite> variable(suiteClassName::suite());                            \
  runner.addTest(variable.get());                                                                      \

#endif /* TEST_HELPER_MACROS_INCLUDED */
