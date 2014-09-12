// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-02-08
 */
//----------------------------------------------------------------------
#include "icl_core_thread/RWLockImplPosix.h"

#include <pthread.h>
#include <icl_core/os_time.h>

#include "icl_core_thread/Common.h"

namespace icl_core {
namespace thread {

RWLockImplPosix::RWLockImplPosix()
  : m_rwlock(NULL)
{
  m_rwlock = new pthread_rwlock_t;
  pthread_rwlock_init(m_rwlock, NULL);
}

RWLockImplPosix::~RWLockImplPosix()
{
  if (m_rwlock)
  {
    pthread_rwlock_destroy(m_rwlock);
    delete m_rwlock;
    m_rwlock = NULL;
  }
}

bool RWLockImplPosix::readLock()
{
  return pthread_rwlock_rdlock(m_rwlock) == 0;
}

// ReadLock with absolute timeout
bool RWLockImplPosix::readLock(const ::icl_core::TimeStamp& timeout)
{
#ifdef _SYSTEM_DARWIN_
  int ret = pthread_rwlock_tryrdlock(m_rwlock);
  while ((ret != 0) && ((timeout > icl_core::TimeStamp::now())))
  {
    // one microsecond
    icl_core::os::usleep(1);
    ret = pthread_rwlock_tryrdlock(m_rwlock);
  }
  return (ret == 0);
#else
  struct timespec timeout_timespec = timeout.timespec();
  int ret = pthread_rwlock_timedrdlock(m_rwlock, &timeout_timespec);
  return (ret == 0);
#endif
}

// ReadLock with relative timeout
bool RWLockImplPosix::readLock(const ::icl_core::TimeSpan& timeout)
{
  return readLock(impl::getAbsoluteTimeout(timeout));
}

bool RWLockImplPosix::tryReadLock()
{
  bool ret = pthread_rwlock_tryrdlock(m_rwlock);
  return (ret == 0);
}

bool RWLockImplPosix::writeLock()
{
  return pthread_rwlock_wrlock(m_rwlock) == 0;
}

// WriteLock with absolute timeout
bool RWLockImplPosix::writeLock(const ::icl_core::TimeStamp& timeout)
{
#ifdef _SYSTEM_DARWIN_
  int ret = pthread_rwlock_trywrlock(m_rwlock);
  while ((ret != 0) && ((timeout > icl_core::TimeStamp::now())))
  {
    // one microsecond
    icl_core::os::usleep(1);
    ret = pthread_rwlock_trywrlock(m_rwlock);
  }
  return (ret == 0);
#else
  struct timespec timeout_timespec = timeout.timespec();
  bool ret = pthread_rwlock_timedwrlock(m_rwlock, &timeout_timespec);
  return (ret == 0);
#endif
}

// WriteLock with relative timeout
bool RWLockImplPosix::writeLock(const ::icl_core::TimeSpan& timeout)
{
  return writeLock(impl::getAbsoluteTimeout(timeout));
}

bool RWLockImplPosix::tryWriteLock()
{
  bool ret = pthread_rwlock_trywrlock(m_rwlock);
  return (ret == 0);
}

void RWLockImplPosix::unlock()
{
  pthread_rwlock_unlock(m_rwlock);
}

}
}
