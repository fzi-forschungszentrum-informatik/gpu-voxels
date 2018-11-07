// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------
#include "ThreadImplLxrt33.h"

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>

#include "Thread.h"
#include "Logging.h"

#define DEFAULT_STACK_SIZE 0x4000

namespace icl_core {
namespace thread {

ThreadImplLxrt33::ThreadImplLxrt33(Thread *thread, const icl_core::String& description,
                                   icl_core::ThreadPriority priority)
  : m_thread_id(0),
    m_thread(thread),
    m_priority(priority),
    m_description(description),
    m_rt_task(NULL)
{ }

ThreadImplLxrt33::~ThreadImplLxrt33()
{ }

void ThreadImplLxrt33::cancel()
{
  if (m_rt_task != NULL)
  {
    rt_task_delete(m_rt_task);
    m_rt_task = NULL;
  }
  pthread_cancel_rt(m_thread_id);
  m_thread_id = 0;
  m_rt_task = NULL;
}

bool ThreadImplLxrt33::isHardRealtime() const
{
  return m_priority < 0;
}

bool ThreadImplLxrt33::executesHardRealtime() const
{
  return os::isThisHRT();
}

void ThreadImplLxrt33::join()
{
  pthread_join_rt(m_thread_id, NULL);
  m_thread_id = 0;
  m_rt_task = NULL;
}

icl_core::ThreadPriority ThreadImplLxrt33::priority() const
{
  return m_priority;
}

bool ThreadImplLxrt33::setHardRealtime(bool hard_realtime)
{
  if (hard_realtime && !os::isThisHRT())
  {
    rt_make_hard_real_time();
    return os::isThisHRT();
  }
  else if (!hard_realtime && os::isThisHRT())
  {
    rt_make_soft_real_time();
    return !os::isThisHRT();
  }
  else
  {
    return false;
  }
}

bool ThreadImplLxrt33::setPriority(icl_core::ThreadPriority priority)
{
  // TODO: Make this work!
  /*
    if (m_rt_task != NULL) {
    int ret = rt_change_prio(m_rt_task, abs(priority));
    if (ret == 0) {
    m_priority = priority;

    if (priority > 0 && IsHardRealtimeThread()) {
    rt_make_soft_real_time();
    } else if (priority < 0 && !IsHardRealtimeThread()) {
    rt_make_hard_real_time();
    }

    return true;
    }
    }
  */

  return false;
}

bool ThreadImplLxrt33::start()
{
  if (pthread_create(&m_thread_id, NULL, ThreadImplLxrt33::runThread, this))
  {
    m_thread_id = 0;
    m_rt_task = NULL;
  }

  return m_thread_id != 0;
}

icl_core::ThreadId ThreadImplLxrt33::threadId() const
{
  return ::icl_core::ThreadId(m_thread_id);
}

void * ThreadImplLxrt33::runThread(void *arg)
{
  ThreadImplLxrt33 *self = static_cast<ThreadImplLxrt33*>(arg);

  self->m_rt_task = rt_task_init(getpid() + pthread_self_rt(),
                                 abs(self->m_priority), DEFAULT_STACK_SIZE, 0);
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
    }

    self->m_thread->runThread();

    rt_make_soft_real_time();
    rt_task_delete(self->m_rt_task);
    self->m_rt_task = NULL;
  }

  return NULL;
}

}
}
