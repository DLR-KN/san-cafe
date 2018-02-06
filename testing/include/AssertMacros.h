/*! \file AssertMacros.h
    \brief Macros to test different types at equality

*/
#ifndef ASSERT_MACROS_INCLUDED
#define ASSERT_MACROS_INCLUDED
#include "TestException.h"
/*!
*  Checks whether two integer values are exactly the same
*/
#define ASSERT_INT_EQUAL(value0, value1)    if(value0 != value1)    \
    {                                                                 \
      throw dlr_test_suite::TestException("INTEGER not equal exception");  \
    }                                                                    \

/*!
*  Checks whether two integer values are different
*/
#define ASSERT_INT_NOT_EQUAL(value0, value1)    if(value0 == value1)    \
{                                                                 \
  throw dlr_test_suite::TestException("INTEGER equal exception");  \
}

/*!
 * Checks whether one int is smaller than the reference value
 */
#define ASSERT_INT_LESS(value0, value1) if(value1 <= value0) \
{\
  throw dlr_test_suite::TestException("INTEGER less than exception");\
}

/*!
*  Checks whether two floating point values are exactly the same
*/
#define ASSERT_FLOAT_EQUAL(value0, value1)    if(value0 != value1)    \
{                                                                 \
  throw dlr_test_suite::TestException("FLOAT not equal exception");  \
}                                                                    \

/*!
*  Checks whether two floating point values are within an epsilon delta
*/
#define ASSERT_FLOAT_ALMOST_EQUAL(value0, value1, epsilon)    if(fabs(value0 - value1) > epsilon)    \
{                                                                 \
  throw dlr_test_suite::TestException("FLOAT delta exceeds epsiolon threshold");  \
}                                                                    \


#endif /*ASSERT_MACROS_INCLUDED*/
