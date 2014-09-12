// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-30
 *
 */
//----------------------------------------------------------------------
#include "ThreadImplLxrt38.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "Thread.h"


namespace icl_core {
namespace logging {

ThreadImplLxrt38::ThreadImplLxrt38(Thread *thread, icl_core::ThreadPriority priority)
  : m_thread_id(0),
    m_thread(thread),
    m_priority(priority),
    m_rt_task(NULL),
    m_rt_start_sync(NULL)
{
}

ThreadImplLxrt38::~ThreadImplLxrt38()
{
  // Ensure that the thread is really destroyed.
  join();

  if (m_rt_start_sync == NULL)
  {
    // Nothing to be done here.
  }
  else
  {
    rt_sem_delete(m_rt_start_sync);
    m_rt_start_sync = NULL;
  }
}

void ThreadImplLxrt38::join()
{
  if (m_thread_id == 0)
  {
    // Nothing to be done here. The thread has already been destroyed.
  }
  else
  {
    pthread_join_rt(m_thread_id, NULL);
    m_rt_task = NULL;
    m_thread_id = 0;
  }
}

bool ThreadImplLxrt38::start()
{
  m_rt_start_sync = rt_typed_sem_init(size_t(this), 2, CNT_SEM | PRIO_Q);
  if (m_rt_start_sync == NULL)
  {
    // We cannot proceed if this happens!
  }
  else
  {
    if (pthread_create(&m_thread_id, NULL, ThreadImplLxrt38::runThread, this))
    {
      m_thread_id = 0;
      m_rt_task = NULL;

      rt_sem_delete(m_rt_start_sync);
      m_rt_start_sync = NULL;
    }
    else
    {
      rt_sem_wait_barrier(m_rt_start_sync);
    }
  }

  return m_thread_id != 0;
}

void *ThreadImplLxrt38::runThread(void *arg)
{
  ThreadImplLxrt38 *self = static_cast<ThreadImplLxrt38*>(arg);

  if (self->m_rt_start_sync == NULL)
  {
    // Technically, this can never happen because this condition is
    // already checked in the Start() function. But who knows!
    PRINTF("ERROR: NULL thread start barrier!\n");
  }
  else
  {
    self->m_rt_task = rt_task_init(getpid() + pthread_self_rt(), abs(self->m_priority),
                                   DEFAULT_STACK_SIZE, 0);
    if (self->m_rt_task == NULL)
    {
      PRINTF("ERROR: Cannot initialize LXRT task %lu!\n", self->m_thread_id);
      PRINTF("       Probably another thread with the same name already exists.\n");

      // Let the thread, which started us, continue!
      rt_sem_wait_barrier(self->m_rt_start_sync);
    }
    else
    {
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
        // This is a soft realtime thread, so nothing additional has
        // to be done here.
      }

      rt_sem_wait_barrier(self->m_rt_start_sync);

      self->m_thread->runThread();

      // Remark: It does not hurt to call this in a soft realtime
      // thread, so just skip the hard realtime test.
      rt_make_soft_real_time();
    }
  }

  return NULL;
}

}
}
