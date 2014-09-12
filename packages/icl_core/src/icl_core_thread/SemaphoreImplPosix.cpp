// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-01-20
 */
//----------------------------------------------------------------------
#include "SemaphoreImplPosix.h"

#include <errno.h>

#include "Common.h"

namespace icl_core {
namespace thread {

SemaphoreImplPosix::SemaphoreImplPosix(size_t initial_value)
  : m_semaphore(0)
{
  m_semaphore = new sem_t;
  sem_init(m_semaphore, PTHREAD_PROCESS_PRIVATE, initial_value);
}

SemaphoreImplPosix::~SemaphoreImplPosix()
{
  if (m_semaphore)
  {
    sem_destroy(m_semaphore);
    delete m_semaphore;
    m_semaphore = 0;
  }
}

void SemaphoreImplPosix::post()
{
  sem_post(m_semaphore);
}

bool SemaphoreImplPosix::tryWait()
{
  int res = sem_trywait(m_semaphore);
  return (res == 0);
}

bool SemaphoreImplPosix::wait()
{
  int res = sem_wait(m_semaphore);
  return (res == 0);
}

bool SemaphoreImplPosix::wait(const icl_core::TimeSpan& timeout)
{
  return wait(impl::getAbsoluteTimeout(timeout));
}

bool SemaphoreImplPosix::wait(const icl_core::TimeStamp& timeout)
{
  struct timespec timeout_spec = timeout.timespec();
  int res = sem_timedwait(m_semaphore, &timeout_spec);
  return (res == 0);
}

}
}
