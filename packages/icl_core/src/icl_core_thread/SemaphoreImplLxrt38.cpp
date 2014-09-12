// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-11-29
 */
//----------------------------------------------------------------------
#include "SemaphoreImplLxrt38.h"

#include <rtai_posix.h>

#include "Common.h"

namespace icl_core {
namespace thread {

SemaphoreImplLxrt38::SemaphoreImplLxrt38(size_t initial_value, int type)
  : m_semaphore(NULL)
{
  m_semaphore = rt_typed_sem_init(size_t(this), initial_value, type | PRIO_Q);
}

SemaphoreImplLxrt38::~SemaphoreImplLxrt38()
{
  if (m_semaphore != NULL)
  {
    rt_sem_delete(m_semaphore);
    m_semaphore = NULL;
  }
}

void SemaphoreImplLxrt38::post()
{
  rt_sem_signal(m_semaphore);
}

bool SemaphoreImplLxrt38::tryWait()
{
  int res = rt_sem_wait_if(m_semaphore);
  return (res > 0 && res < SEM_TIMOUT);
}

bool SemaphoreImplLxrt38::wait()
{
  int res = rt_sem_wait(m_semaphore);
  return (res < SEM_TIMOUT);
}

bool SemaphoreImplLxrt38::wait(const icl_core::TimeSpan& timeout)
{
  return wait(impl::getAbsoluteTimeout(timeout));
}

bool SemaphoreImplLxrt38::wait(const icl_core::TimeStamp& timeout)
{
  struct timespec timeout_spec = timeout.systemTimespec();
  int res = rt_sem_wait_until(m_semaphore, timespec2count(&timeout_spec));
  return (res < SEM_TIMOUT);
}

}
}
