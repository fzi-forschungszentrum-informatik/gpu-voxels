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
 */
//----------------------------------------------------------------------
#include <iostream>
#include <iomanip>

#include "icl_core_testsuite/TestListener.h"

namespace icl_core {

void TestListener::startTest(CPPUNIT_NS::Test *test)
{
  std::cerr << "  Running " << std::setw(60) << std::left << test->getName();
  m_success = true;
}

void TestListener::addFailure(const CPPUNIT_NS::TestFailure&)
{
  m_success = false;
}

void TestListener::endTest(CPPUNIT_NS::Test*)
{
  std::cerr << (m_success ? "OK" : "FAILED") << std::endl;
}

void TestListener::startSuite(CPPUNIT_NS::Test *test)
{
  std::cerr << "Running test suite " << test->getName() << ":" << std::endl;
}

void TestListener::endSuite(CPPUNIT_NS::Test *test)
{
  std::cerr << "Test suite " << test->getName() << " finished." << std::endl;
}

}
