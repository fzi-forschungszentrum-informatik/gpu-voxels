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
#include "ThreadImplLxrt35.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "Thread.h"

namespace icl_core {
namespace logging {

ThreadImplLxrt35::ThreadImplLxrt35(Thread *thread, icl_core::ThreadPriority priority)
  : m_thread_id(0),
    m_thread(thread),
    m_priority(priority),
    m_rt_task(NULL),
    m_rt_start_sync(NULL)
{ }

ThreadImplLxrt35::~ThreadImplLxrt35()
{
  // Ensure that the thread is really destroyed.
  join();

  if (m_rt_start_sync == NULL)
  {
    // Nothing to be done here.
  }
  else
  {
    pthread_barrier_destroy_rt(m_rt_start_sync);
    delete m_rt_start_sync;
    m_rt_start_sync = NULL;
  }
}

void ThreadImplLxrt35::join()
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

bool ThreadImplLxrt35::start()
{
  m_rt_start_sync = new (std::nothrow) pthread_barrier_t;
  if (m_rt_start_sync == NULL)
  {
    // We cannot proceed, if this happens!
  }
  else
  {
    pthread_barrier_init_rt(m_rt_start_sync, NULL, 2);

    if (pthread_create(&m_thread_id, NULL, ThreadImplLxrt35::runThread, this))
    {
      m_thread_id = 0;
      m_rt_task = NULL;

      pthread_barrier_destroy_rt(m_rt_start_sync);
      delete m_rt_start_sync;
      m_rt_start_sync = NULL;
    }
    else
    {
      pthread_barrier_wait_rt(m_rt_start_sync);
    }
  }

  return m_thread_id != 0;
}

void *ThreadImplLxrt35::runThread(void *arg)
{
  ThreadImplLxrt35 *self = static_cast<ThreadImplLxrt35*>(arg);

  if (self->m_rt_start_sync == NULL)
  {
    // Technically, this can never happen because this condition is already checked
    // in the Start() function. But who knows!
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
      pthread_barrier_wait_rt(self->m_rt_start_sync);
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

      pthread_barrier_wait_rt(self->m_rt_start_sync);

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
