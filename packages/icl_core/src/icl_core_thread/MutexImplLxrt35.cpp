// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-06-09
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2009-11-09
 *
 */
//----------------------------------------------------------------------
#include "icl_core_thread/MutexImplLxrt35.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "icl_core_thread/Common.h"

#define STRICT_LXRT_CHECKS


namespace icl_core {
namespace thread {

MutexImplLxrt35::MutexImplLxrt35()
  : m_mutex(0)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::MutexImplLxrt35: Called from a Linux task!\n");
    return;
  }
#endif
  m_mutex = new pthread_mutex_t;
  pthread_mutex_init_rt(m_mutex, NULL);
}

MutexImplLxrt35::~MutexImplLxrt35()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::~MutexImplLxrt35: Called from a Linux task!\n");
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

bool MutexImplLxrt35::lock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::lock: Called from a Linux task!\n");
    return false;
  }
#endif
  return pthread_mutex_lock_rt(m_mutex) == 0;
}

bool MutexImplLxrt35::lock(const icl_core::TimeSpan& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::lock: Called from a Linux task!\n");
    return false;
  }
#endif
  return lock(impl::getAbsoluteTimeout(timeout));
}

bool MutexImplLxrt35::lock(const icl_core::TimeStamp& timeout)
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::lock: Called from a Linux task!\n");
    return false;
  }
#endif
  struct timespec timeout_absolute_timespec = timeout.systemTimespec();
  bool ret = (pthread_mutex_timedlock_rt(m_mutex, & timeout_absolute_timespec) == 0);
  return ret;
}

bool MutexImplLxrt35::tryLock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::tryLock: Called from a Linux task!\n");
    return false;
  }
#endif
  bool ret = (pthread_mutex_trylock_rt(m_mutex) == 0);
  return ret;
}

void MutexImplLxrt35::unlock()
{
#ifdef STRICT_LXRT_CHECKS
  if (!icl_core::os::isThisLxrtTask())
  {
    PRINTF("MutexImplLxrt35::unlock: Called from a Linux task!\n");
    return;
  }
#endif
  pthread_mutex_unlock_rt(m_mutex);
}

}
}
