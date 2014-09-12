// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-03-02
 *
 */
//----------------------------------------------------------------------
#include "ThreadImplLxrt33.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "Thread.h"

namespace icl_core {
namespace logging {

ThreadImplLxrt33::ThreadImplLxrt33(Thread *thread, icl_core::ThreadPriority priority)
  : m_thread_id(0),
    m_thread(thread),
    m_priority(priority),
    m_rt_task(NULL)
{
}

ThreadImplLxrt33::~ThreadImplLxrt33()
{
}

void ThreadImplLxrt33::join()
{
  pthread_join_rt(m_thread_id, NULL);
  m_rt_task = NULL;
}

bool ThreadImplLxrt33::start()
{
  if (pthread_create(&m_thread_id, NULL, ThreadImplLxrt33::runThread, this))
  {
    m_thread_id = 0;
    m_rt_task = NULL;
  }
  else
  {
    // Nothing to be done here!
  }

  return m_thread_id != 0;
}

void *ThreadImplLxrt33::runThread(void *arg)
{
  ThreadImplLxrt33 *self = static_cast<ThreadImplLxrt33*>(arg);

  self->m_rt_task = rt_task_init(getpid() + pthread_self_rt(), abs(self->m_priority),
                                 DEFAULT_STACK_SIZE, 0);
  if (self->m_rt_task == NULL)
  {
    PRINTF("ERROR: Cannot initialize LXRT task %lu!\n", self->m_thread_id);
    PRINTF("       Probably another thread with the same name already exists.\n");
  }
  else
  {
    rt_task_use_fpu(self->m_rt_task, 1);

    if (self->m_priority < 0)
    {
      rt_make_hard_real_time();
      if (!rt_is_hard_real_time(rt_buddy()))
      {
        PRINTF("ERROR: Setting thread %lu to hard real-time failed!\n", self->m_thread_id);
      }
      else
      {
        // Everything worked as expected, so no message here.
      }
    }
    else
    {
      // This is a soft realtime thread, so nothing additional has to
      // be done here.
    }

    self->m_thread->runThread();

    rt_make_soft_real_time();

    // TODO: Check if this is correct. The RTAI 3.5 and 3.8
    // implementations leave this to a call to join().
    rt_task_delete(self->m_rt_task);
    self->m_rt_task = NULL;
  }

  return NULL;
}

}
}
