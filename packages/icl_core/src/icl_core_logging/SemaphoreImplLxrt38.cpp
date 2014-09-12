// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-30
 */
//----------------------------------------------------------------------
#include "SemaphoreImplLxrt38.h"

namespace icl_core {
namespace logging {

SemaphoreImplLxrt38::SemaphoreImplLxrt38(size_t initial_value)
  : m_semaphore(NULL)
{
  m_semaphore = rt_typed_sem_init(size_t(this), initial_value, CNT_SEM | PRIO_Q);
}

SemaphoreImplLxrt38::~SemaphoreImplLxrt38()
{
  if (m_semaphore == NULL)
  {
    // Nothing to be done here!
  }
  else
  {
    rt_sem_delete(m_semaphore);
    m_semaphore = NULL;
  }
}

void SemaphoreImplLxrt38::post()
{
  rt_sem_signal(m_semaphore);
}

bool SemaphoreImplLxrt38::wait()
{
  int res = rt_sem_wait(m_semaphore);
  return (res < SEM_TIMOUT);
}

}
}
