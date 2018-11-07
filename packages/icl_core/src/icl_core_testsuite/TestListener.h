// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-01-06
 *
 * \brief   Contains icl_core::TestListener
 *
 * \b icl_core::TestListener
 *
 * A few words for icl_core::TestListener
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_TESTSUITE_TEST_LISTENER_H_INCLUDED
#define ICL_CORE_TESTSUITE_TEST_LISTENER_H_INCLUDED

#include <cppunit/Test.h>
#include <cppunit/TestFailure.h>
#include <cppunit/TestListener.h>

namespace icl_core {

/*!
 * Outputs status lines for each test case.
 */
class TestListener : public CPPUNIT_NS::TestListener
{
public:
  virtual ~TestListener() {}

  virtual void startTest(CPPUNIT_NS::Test *test);
  virtual void addFailure(const CPPUNIT_NS::TestFailure& failure);
  virtual void endTest(CPPUNIT_NS::Test *test);

  virtual void startSuite(CPPUNIT_NS::Test *test);
  virtual void endSuite(CPPUNIT_NS::Test *test);

private:
  bool m_success;
};

}

#endif
