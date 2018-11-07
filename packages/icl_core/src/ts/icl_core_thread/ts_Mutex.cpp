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
 * \date    2009-12-14
 *
 */
//----------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include <icl_core_thread/Mutex.h>
#include <icl_core_thread/Thread.h>

using icl_core::TimeSpan;
using icl_core::TimeStamp;
using icl_core::thread::Mutex;
using icl_core::thread::Thread;

const TimeSpan timeout(1, 0);

class MutexTestThread : public Thread
{
public:
  MutexTestThread(Mutex *mutex) :
    Thread("Mutex Test Thread"),
    m_has_run(false),
    m_mutex(mutex)
  {
  }

  virtual ~MutexTestThread()
  {
  }

  virtual void run()
  {
    BOOST_CHECK(!m_mutex->tryLock());
    BOOST_CHECK(!m_mutex->lock(timeout));
    BOOST_CHECK(!m_mutex->lock(TimeStamp::now() + timeout));

    m_has_run = true;
  }

  bool hasRun() const { return m_has_run; }

private:
  bool m_has_run;
  Mutex *m_mutex;
};

BOOST_AUTO_TEST_SUITE(ts_Mutex)

BOOST_AUTO_TEST_CASE(MutexLock)
{
  Mutex mutex;
  BOOST_CHECK(mutex.lock());
  mutex.unlock();
}

BOOST_AUTO_TEST_CASE(MutexTryLock)
{
  Mutex mutex;
  BOOST_CHECK(mutex.tryLock());
  mutex.unlock();
}

BOOST_AUTO_TEST_CASE(MutexLockAbsoluteTimeout)
{
  Mutex mutex;
  BOOST_CHECK(mutex.lock(TimeStamp::now() + TimeSpan(1, 0)));
  mutex.unlock();
}

BOOST_AUTO_TEST_CASE(MutexLockRelativeTimeout)
{
  Mutex mutex;
  BOOST_CHECK(mutex.lock(TimeSpan(1, 0)));
  mutex.unlock();
}

BOOST_AUTO_TEST_CASE(MultiThreadMutexTest)
{
  Mutex mutex;

  BOOST_CHECK(mutex.lock());

  MutexTestThread *thread = new MutexTestThread(&mutex);

  thread->start();
  thread->join();
  BOOST_CHECK(thread->hasRun());

  delete thread;

  mutex.unlock();
}

BOOST_AUTO_TEST_SUITE_END()
