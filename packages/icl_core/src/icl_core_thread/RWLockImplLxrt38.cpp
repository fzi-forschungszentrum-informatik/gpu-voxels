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
#include "icl_core_thread/RWLockImplLxrt38.h"

#include <rtai_posix.h>

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "icl_core_thread/Common.h"

#undef STRICT_LXRT_CHECKS


namespace icl_core {
namespace thread {

RWLockImplLxrt38::RWLockImplLxrt38()
  : m_rwlock(NULL)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::RWLockImplLxrt38: Called from a Linux task!\n");
    return;
  }
#endif
  m_rwlock = rt_typed_rwl_init(size_t(this), RESEM_RECURS);
}

RWLockImplLxrt38::~RWLockImplLxrt38()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::~RWLockImplLxrt38: Called from a Linux task!\n");
    return;
  }
#endif
  if (m_rwlock)
  {
    rt_rwl_delete(m_rwlock);
    m_rwlock = NULL;
  }
}

bool RWLockImplLxrt38::readLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::readLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return rt_rwl_rdlock(m_rwlock) == 0;
}

bool RWLockImplLxrt38::readLock(const icl_core::TimeStamp& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::readLock: Called from a Linux task!\n");
    return false;
  }
#endif
  struct timespec timeout_spec = timeout.systemTimespec();
  int ret = rt_rwl_rdlock_until(m_rwlock, timespec2count(&timeout_spec));
  return (ret == 0);
}

bool RWLockImplLxrt38::readLock(const icl_core::TimeSpan& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::readLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return readLock(impl::getAbsoluteTimeout(timeout));
}

bool RWLockImplLxrt38::tryReadLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::tryReadLock: Called from a Linux task!\n");
    return false;
  }
#endif
  int ret = rt_rwl_rdlock_if(m_rwlock);
  return (ret == 0);
}

bool RWLockImplLxrt38::writeLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::writeLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return rt_rwl_wrlock(m_rwlock) == 0;
}

bool RWLockImplLxrt38::writeLock(const icl_core::TimeStamp& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::writeLock: Called from a Linux task!\n");
    return false;
  }
#endif
  struct timespec timeout_spec = timeout.systemTimespec();
  int ret = rt_rwl_wrlock_until(m_rwlock, timespec2count(&timeout_spec));
  return (ret == 0);
}

bool RWLockImplLxrt38::writeLock(const icl_core::TimeSpan& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::writeLock: Called from a Linux task!\n");
    return false;
  }
#endif
  return writeLock(impl::getAbsoluteTimeout(timeout));
}

bool RWLockImplLxrt38::tryWriteLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::tryWriteLock: Called from a Linux task!\n");
    return false;
  }
#endif
  int ret = rt_rwl_wrlock_if(m_rwlock);
  return ret == 0;
  /*
  // ATTENTION: Calling pthread_rwlock_trywrlock_rt() while another
  // thread holds a read lock seems to be buggy in RTAI 3.3, so the
  // following does NOT work:
  //   int ret = pthread_rwlock_trywrlock_rt(rwlock);
  //   return (ret == 0);
  // Therefore we call WriteLock() with a very short timeout!
  static icl_core::TimeSpan try_write_lock_timeout(0, 1);
  return writeLock(try_write_lock_timeout);
  */
}

void RWLockImplLxrt38::unlock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("RWLockImplLxrt38::unlock: Called from a Linux task!\n");
    return;
  }
#endif
  rt_rwl_unlock(m_rwlock);
}

}
}
