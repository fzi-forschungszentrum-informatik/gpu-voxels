// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------
#include "icl_core_thread/MutexImplPosix.h"

#include <pthread.h>
#include <icl_core/os_time.h>

#include "icl_core_thread/Common.h"

namespace icl_core {
namespace thread {

MutexImplPosix::MutexImplPosix()
  : m_mutex(NULL)
{
  m_mutex = new pthread_mutex_t;
  pthread_mutexattr_t mutex_attr;
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(m_mutex, &mutex_attr);
  pthread_mutexattr_destroy(&mutex_attr);
}

MutexImplPosix::~MutexImplPosix()
{
  if (m_mutex)
  {
    pthread_mutex_destroy(m_mutex);
    delete m_mutex;
    m_mutex = NULL;
  }
}

bool MutexImplPosix::lock()
{
  return pthread_mutex_lock(m_mutex) == 0;
}

bool MutexImplPosix::lock(const ::icl_core::TimeStamp& timeout)
{
#ifdef _SYSTEM_DARWIN_
  int ret = pthread_mutex_trylock(m_mutex);
  while ((ret != 0) && ((timeout > icl_core::TimeStamp::now())))
  {
    // one microsecond
    icl_core::os::usleep(1);
    ret = pthread_mutex_trylock(m_mutex);
  }
  return ret == 0;
#else
  struct timespec timeout_spec = timeout.timespec();
  int ret = pthread_mutex_timedlock(m_mutex, &timeout_spec);
  return (ret == 0);
#endif
}

bool MutexImplPosix::lock(const ::icl_core::TimeSpan& timeout)
{
  return lock(impl::getAbsoluteTimeout(timeout));
}

bool MutexImplPosix::tryLock()
{
  int ret = pthread_mutex_trylock(m_mutex);
  return (ret == 0);
}

void MutexImplPosix::unlock()
{
  pthread_mutex_unlock(m_mutex);
}

}
}
