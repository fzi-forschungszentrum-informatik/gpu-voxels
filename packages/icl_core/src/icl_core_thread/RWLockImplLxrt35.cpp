// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-30
 *
 */
//----------------------------------------------------------------------
#include "icl_core_thread/RWLockImplLxrt35.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "icl_core_thread/Common.h"

#undef STRICT_LXRT_CHECKS


namespace icl_core {
namespace thread {

RWLockImplLxrt35::RWLockImplLxrt35()
  : m_rwlock(NULL)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::RWLockImplLxrt35: Called from a Linux task!\n");
    return;
  }
#endif
  m_rwlock = new pthread_rwlock_t;
  pthread_rwlock_init_rt(m_rwlock, NULL);
}

RWLockImplLxrt35::~RWLockImplLxrt35()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::~RWLockImplLxrt35: Called from a Linux task!\n");
    return;
  }
#endif
  if (m_rwlock)
  {
    pthread_rwlock_destroy_rt(m_rwlock);
    delete m_rwlock;
    m_rwlock = NULL;
  }
}

bool RWLockImplLxrt35::readLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::readLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return pthread_rwlock_rdlock_rt(m_rwlock) == 0;
}

bool RWLockImplLxrt35::readLock(const icl_core::TimeStamp& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::readLock: Called from a Linux task!\n");
    return false;
  }
#endif
  struct timespec timeout_absolute_timespec = timeout.systemTimespec();
  int ret = pthread_rwlock_timedrdlock_rt(m_rwlock, &timeout_absolute_timespec);
  return (ret == 0);
}

bool RWLockImplLxrt35::readLock(const icl_core::TimeSpan& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::readLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return readLock(impl::getAbsoluteTimeout(timeout));
}

bool RWLockImplLxrt35::tryReadLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::tryReadLock: Called from a Linux task!\n");
    return false;
  }
#endif
  int ret = pthread_rwlock_tryrdlock_rt(m_rwlock);
  return (ret == 0);
}

bool RWLockImplLxrt35::writeLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::writeLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return pthread_rwlock_wrlock_rt(m_rwlock) == 0;
}

bool RWLockImplLxrt35::writeLock(const icl_core::TimeStamp& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::writeLock: Called from a Linux task!\n");
    return false;
  }
#endif
  struct timespec timeout_absolute_timespec = timeout.systemTimespec();
  int ret = pthread_rwlock_timedwrlock_rt(m_rwlock, &timeout_absolute_timespec);
  return (ret == 0);
}

bool RWLockImplLxrt35::writeLock(const icl_core::TimeSpan& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::writeLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return writeLock(impl::getAbsoluteTimeout(timeout));
}

bool RWLockImplLxrt35::tryWriteLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::tryWriteLock: Called from a Linux task!\n");
    return false;
  }
#endif
  // ATTENTION: Calling pthread_rwlock_trywrlock_rt() while another
  // thread holds a read lock seems to be buggy in RTAI 3.3, so the
  // following does NOT work:
  // int ret = pthread_rwlock_trywrlock_rt(rwlock);
  // return (ret == 0);
  // Therefore we call WriteLock() with a very short timeout!
  static icl_core::TimeSpan try_write_lock_timeout(0, 1);
  return writeLock(try_write_lock_timeout);
}

void RWLockImplLxrt35::unlock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt35::unlock: Called from a Linux task!\n");
    return;
  }
#endif
  pthread_rwlock_unlock_rt(rwlock);
}

}
}
