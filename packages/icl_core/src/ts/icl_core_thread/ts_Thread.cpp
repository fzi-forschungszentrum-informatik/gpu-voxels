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
 * \date    2009-12-30
 *
 */
//----------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <icl_core_thread/Thread.h>

using icl_core::thread::Thread;

class TestThread : public Thread
{
public:
  TestThread()
    : Thread("Test Thread"),
      m_has_run(false)
  { }

  virtual ~TestThread()
  { }

  virtual void run()
  {
    m_has_run = true;
  }

  bool hasRun() const { return m_has_run; }

private:
  bool m_has_run;
};

BOOST_AUTO_TEST_SUITE(ts_Thread)

BOOST_AUTO_TEST_CASE(RunThread)
{
  TestThread test_thread;

  BOOST_CHECK(!test_thread.hasRun());

  test_thread.start();
  test_thread.stop();
  test_thread.join();

  BOOST_CHECK(test_thread.hasRun());
}

BOOST_AUTO_TEST_SUITE_END()
