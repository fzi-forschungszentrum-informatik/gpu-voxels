// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-06-09
 *
 */
//----------------------------------------------------------------------
#include "icl_core_thread/MutexImplLxrt33.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "icl_core_thread/Common.h"

#undef STRICT_LXRT_CHECKS

namespace icl_core {
namespace thread {

MutexImplLxrt33::MutexImplLxrt33()
  : m_mutex(0)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::IsThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::MutexImplLxrt33: Called from a Linux task!\n");
    return;
  }
#endif
  m_mutex = new pthread_mutex_t;
  pthread_mutex_init_rt(m_mutex, NULL);
}

MutexImplLxrt33::~MutexImplLxrt33()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::IsThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::~MutexImplLxrt33: Called from a Linux task!\n");
    return;
  }
#endif
  if (m_mutex)
  {
    pthread_mutex_destroy_rt(m_mutex);
    delete m_mutex;
    m_mutex = 0;
  }
}

bool MutexImplLxrt33::lock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::lock: Called from a Linux task!\n");
    return false;
  }
#endif
  return pthread_mutex_lock_rt(m_mutex) == 0;
}

bool MutexImplLxrt33::lock(const icl_core::TimeSpan& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::lock: Called from a Linux task!\n");
    return false;
  }
#endif
  return lock(impl::getAbsoluteTimeout(timeout));
}

bool MutexImplLxrt33::lock(const icl_core::TimeStamp& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::lock: Called from a Linux task!\n");
    return false;
  }
#endif
  struct timespec timeout_absolute_timespec = timeout.systemTimespec();
  bool ret = (pthread_mutex_timedlock_rt(m_mutex, & timeout_absolute_timespec) == 0);
  return ret;
}

bool MutexImplLxrt33::tryLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::tryLock: Called from a Linux task!\n");
    return false;
  }
#endif
  bool ret = (pthread_mutex_trylock_rt(m_mutex) == 0);
  return ret;
}

void MutexImplLxrt33::unlock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt33::unlock: Called from a Linux task!\n");
    return;
  }
#endif
  pthread_mutex_unlock_rt(m_mutex);
}

}
}
