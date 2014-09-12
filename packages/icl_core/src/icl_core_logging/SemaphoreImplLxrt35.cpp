// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-30
 */
//----------------------------------------------------------------------
#include "SemaphoreImplLxrt35.h"

#include <errno.h>

namespace icl_core {
namespace logging {

SemaphoreImplLxrt35::SemaphoreImplLxrt35(size_t initial_value)
  : m_semaphore(NULL)
{
  m_semaphore = new sem_t;
  sem_init_rt(m_semaphore, PTHREAD_PROCESS_PRIVATE, initial_value);
}

SemaphoreImplLxrt35::~SemaphoreImplLxrt35()
{
  if (m_semaphore == NULL)
  {
    // Nothing to be done here!
  }
  else
  {
    sem_destroy_rt(m_semaphore);
    delete m_semaphore;
    m_semaphore = NULL;
  }
}

void SemaphoreImplLxrt35::post()
{
  sem_post_rt(m_semaphore);
}

bool SemaphoreImplLxrt35::wait()
{
  int res = sem_wait_rt(m_semaphore);
  return (res == 0);
}

}
}
