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
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \date    2010-02-08
 *
 */
//----------------------------------------------------------------------
#include <boost/test/unit_test.hpp>
#include "icl_core_thread/Thread.h"
#include "icl_core_thread/RWLock.h"
#include "icl_core/os_time.h"

using icl_core::TimeSpan;
using icl_core::TimeStamp;
using icl_core::thread::RWLock;
using icl_core::thread::Mutex;
using icl_core::thread::Thread;


class RWLockTestThread : public Thread
{
public:
  RWLockTestThread(RWLock *rwlock, bool read_lock)
    : Thread("Test Thread"),
      m_has_run(false),
      m_read_lock(read_lock),
      m_rwlock(rwlock)
  { }

  virtual ~RWLockTestThread()
  { }

  virtual void run()
  {
    if (m_read_lock)
    {
      BOOST_CHECK(m_rwlock->tryReadLock());
      BOOST_CHECK(m_rwlock->readLock(icl_core::TimeSpan(1, 0)));
      BOOST_CHECK(m_rwlock->readLock(icl_core::TimeStamp::now() + icl_core::TimeSpan(1, 0)));
    }
    else
    {
      BOOST_CHECK(!m_rwlock->tryReadLock());
      BOOST_CHECK(!m_rwlock->readLock(icl_core::TimeSpan(1, 0)));
      BOOST_CHECK(!m_rwlock->readLock(icl_core::TimeStamp::now() + icl_core::TimeSpan(1, 0)));
    }

    BOOST_CHECK(!m_rwlock->tryWriteLock());
    BOOST_CHECK(!m_rwlock->writeLock(icl_core::TimeSpan(1, 0)));
    BOOST_CHECK(!m_rwlock->writeLock(icl_core::TimeStamp::now() + icl_core::TimeSpan(1, 0)));

    m_has_run = true;
  }

  bool hasRun() const { return m_has_run; }

private:
  bool m_has_run;
  bool m_read_lock;
  RWLock *m_rwlock;
};


BOOST_AUTO_TEST_SUITE(ts_RWLock)

BOOST_AUTO_TEST_CASE(ReadLock)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.readLock());
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(TryReadLock)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.tryReadLock());
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(ReadLockAbsoluteTimeout)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.readLock(TimeStamp::now() + TimeSpan(1, 0)));
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(ReadLockRelativeTimeout)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.readLock(TimeSpan(1, 0)));
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(WriteLock)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.writeLock());
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(TryWriteLock)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.tryWriteLock());
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(WriteLockAbsoluteTimeout)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.writeLock(TimeStamp::now() + TimeSpan(1, 0)));
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(WriteLockRelativeTimeout)
{
  RWLock rwlock;
  BOOST_CHECK(rwlock.writeLock(TimeSpan(1, 0)));
  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(ReadAndWriteLockWhileReadLock)
{
  RWLock rwlock;

  BOOST_CHECK(rwlock.readLock());

  RWLockTestThread *testthread = new RWLockTestThread(&rwlock, true);

  testthread->start();
  testthread->join();
  BOOST_CHECK(testthread->hasRun());

  delete testthread;

  rwlock.unlock();
}

BOOST_AUTO_TEST_CASE(ReadAndWriteLockWhileWriteLock)
{
  RWLock rwlock;

  BOOST_CHECK(rwlock.writeLock());

  RWLockTestThread *testthread = new RWLockTestThread(&rwlock, false);

  testthread->start();
  testthread->join();
  BOOST_CHECK(testthread->hasRun());

  delete testthread;

  rwlock.unlock();
}

BOOST_AUTO_TEST_SUITE_END()
