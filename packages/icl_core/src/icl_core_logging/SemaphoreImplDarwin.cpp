// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-02-02
 */
//----------------------------------------------------------------------
#include <mach/mach_init.h>
#include <mach/task.h>

#include "SemaphoreImplDarwin.h"

namespace icl_core {
namespace logging {

SemaphoreImplDarwin::SemaphoreImplDarwin(size_t initial_value)
  : m_semaphore(0)
{
  semaphore_create(mach_task_self(), &m_semaphore, SYNC_POLICY_FIFO, initial_value);
}

SemaphoreImplDarwin::~SemaphoreImplDarwin()
{
  semaphore_destroy(mach_task_self(), m_semaphore);
  m_semaphore = 0;
}

void SemaphoreImplDarwin::post()
{
  semaphore_signal(m_semaphore);
}

bool SemaphoreImplDarwin::wait()
{
  kern_return_t res = semaphore_wait(m_semaphore);
  return (res == KERN_SUCCESS);
}

}
}
