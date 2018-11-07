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
#include <icl_core/BaseTypes.h>
#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>
#include <icl_core/TimeSpan.h>
#include <icl_core/TimeStamp.h>
#include <icl_core_thread/PeriodicThread.h>

using icl_core::TimeSpan;
using icl_core::TimeStamp;
using icl_core::thread::PeriodicThread;

BOOST_AUTO_TEST_SUITE(ts_PeriodicThread)

const icl_core::TimeSpan cBURN_THREAD_PERIOD(0, 100000000);

#ifdef _SYSTEM_LXRT_
const icl_core::ThreadPriority cTEST_THREAD_PRIORITY = -10;
const icl_core::ThreadPriority cBURN_THREAD_PRIORITY = 8;
const icl_core::ThreadPriority cRUN_THREAD_PRIORITY = 19;
const double cMAX_DEVIATION_FACTOR = 0.05;
const double cMEAN_DEVIATION_FACTOR = 0.02;
// Attention: Don't increase this beyond 9 (90% CPU time), because
// the test-suite will not terminate otherwise!
const size_t cNUMBER_OF_BURN_THREADS = 9;

# define RUN_HARD_REALTIME_TESTS

#else
const icl_core::ThreadPriority cTEST_THREAD_PRIORITY = 19;
const icl_core::ThreadPriority cBURN_THREAD_PRIORITY = 8;
const icl_core::ThreadPriority cRUN_THREAD_PRIORITY = 18;
const double cMAX_DEVIATION_FACTOR = 1;
const double cMEAN_DEVIATION_FACTOR = 1;
const size_t cNUMBER_OF_BURN_THREADS = 10;
#endif

/*! Thread for testing how exact the periods are executed.
 */
class PeriodicTestThread : public PeriodicThread
{
public:
  PeriodicTestThread(const TimeSpan& period, size_t runs)
    : PeriodicThread("Test Thread", period, cTEST_THREAD_PRIORITY),
      m_has_run(false),
      m_runs(runs)
  { }
  virtual ~PeriodicTestThread() {}

  virtual void run()
  {
    m_has_run = true;

    // Wait for the first period so that the timing is in sync.
    waitPeriod();

    TimeStamp last_run = TimeStamp::now();
    for (size_t run = 0; run < m_runs; ++run)
    {
      waitPeriod();

      TimeStamp now = TimeStamp::now();
      TimeSpan deviation = abs(now - last_run - period());
      if (deviation > m_max_deviation)
      {
        m_max_deviation = deviation;
      }
      m_accumulated_deviation += deviation;

      last_run = now;
    }
  }

  bool hasRun() const { return m_has_run; }
  TimeSpan maxDeviation() const { return m_max_deviation; }
  TimeSpan accumulatedDeviation() const { return m_accumulated_deviation; }
  TimeSpan meanDeviation() const { return m_accumulated_deviation / m_runs; }

private:
  bool m_has_run;
  size_t m_runs;
  TimeSpan m_max_deviation;
  TimeSpan m_accumulated_deviation;
};

/*! Thread for burning away 10% of CPU time.
 */
class BurnThread : public icl_core::thread::PeriodicThread,
                   virtual protected icl_core::Noncopyable
{
public:
  BurnThread(size_t num)
    : PeriodicThread("Burn Thread", cBURN_THREAD_PERIOD, cBURN_THREAD_PRIORITY),
      m_num(num)
  { }

  virtual ~BurnThread()
  { }

  virtual void run()
  {
    while (execute())
    {
      waitPeriod();

      icl_core::TimeStamp now = icl_core::TimeStamp::now();

      // Burn 10% CPU time.
      icl_core::TimeStamp burn_until = now + cBURN_THREAD_PERIOD * 0.1;
      while (icl_core::TimeStamp::now() < burn_until)
      {
        // Just do nothing ;-)
      }
    }
  }

private:
  size_t m_num;
};

void runPeriodicThread(const TimeSpan& period, size_t runs,
                       const TimeSpan& max_deviation, const TimeSpan& mean_deviation,
                       bool burn = false)
{
  PeriodicTestThread test_thread(period, runs);
  BurnThread *burn_threads[cNUMBER_OF_BURN_THREADS];
  memset(burn_threads, 0, sizeof(burn_threads));
  if (burn)
  {
    for (size_t i = 0; i < cNUMBER_OF_BURN_THREADS; ++i)
    {
      burn_threads[i] = new BurnThread(i);
    }
  }

  BOOST_CHECK(!test_thread.hasRun());

  test_thread.start();
  test_thread.stop();

  if (burn)
  {
    for (size_t i = 0; i < cNUMBER_OF_BURN_THREADS; ++i)
    {
      burn_threads[i]->start();
    }
  }

  test_thread.join();

  if (burn)
  {
    for (size_t i = 0; i < cNUMBER_OF_BURN_THREADS; ++i)
    {
      burn_threads[i]->stop();
      burn_threads[i]->join();
      delete burn_threads[i];
      burn_threads[i] = NULL;
    }
  }

  BOOST_REQUIRE(test_thread.hasRun());
  BOOST_TEST_MESSAGE("max deviation=" << test_thread.maxDeviation().toNSec() << "ns" <<
                     ", accumulated deviation=" << test_thread.accumulatedDeviation().toNSec() << "ns" <<
                     ", mean deviation=" << test_thread.meanDeviation().toNSec() << "ns");
  BOOST_CHECK(test_thread.maxDeviation() < max_deviation);
  BOOST_CHECK(test_thread.meanDeviation() < mean_deviation);
}

BOOST_AUTO_TEST_CASE(RunPeriodicThread_1s)
{
  TimeSpan period(1, 0);
  runPeriodicThread(period, 10, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR);
}

BOOST_AUTO_TEST_CASE(RunPeriodicThread_100ms)
{
  TimeSpan period(0, 100000000);
  runPeriodicThread(period, 100, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR);
}

#ifdef RUN_HARD_REALTIME_TESTS

BOOST_AUTO_TEST_CASE(RunPeriodicThread_10ms)
{
  TimeSpan period(0, 10000000);
  runPeriodicThread(period, 1000, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR);
}

BOOST_AUTO_TEST_CASE(RunPeriodicThread_1ms)
{
  TimeSpan period(0, 1000000);
  runPeriodicThread(period, 10000, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR);
}

BOOST_AUTO_TEST_CASE(BurnPeriodicThread_1s)
{
  TimeSpan period(1, 0);
  runPeriodicThread(period, 10, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR, true);
}

BOOST_AUTO_TEST_CASE(BurnPeriodicThread_100ms)
{
  TimeSpan period(0, 100000000);
  runPeriodicThread(period, 100, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR, true);
}

BOOST_AUTO_TEST_CASE(BurnPeriodicThread_10ms)
{
  TimeSpan period(0, 10000000);
  runPeriodicThread(period, 1000, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR, true);
}

BOOST_AUTO_TEST_CASE(BurnPeriodicThread_1ms)
{
  TimeSpan period(0, 1000000);
  runPeriodicThread(period, 10000, period * cMAX_DEVIATION_FACTOR, period * cMEAN_DEVIATION_FACTOR, true);
}

#endif

BOOST_AUTO_TEST_SUITE_END()
