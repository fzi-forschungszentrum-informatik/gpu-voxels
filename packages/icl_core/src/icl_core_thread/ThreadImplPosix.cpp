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
#include "ThreadImplPosix.h"

#include <assert.h>
#include <pthread.h>

#include <icl_core/os_posix_thread.h>

#include "Thread.h"

namespace icl_core {
namespace thread {

ThreadImplPosix::ThreadImplPosix(Thread *thread, const icl_core::String& description,
                                 icl_core::ThreadPriority priority)
  : m_thread_id(0),
    m_thread(thread),
    m_priority(priority),
    m_description(description)
{ }

ThreadImplPosix::~ThreadImplPosix()
{ }

void ThreadImplPosix::cancel()
{
  pthread_cancel(m_thread_id);
}

void ThreadImplPosix::join()
{
  pthread_join(m_thread_id, NULL);
}

icl_core::ThreadPriority ThreadImplPosix::priority() const
{
  struct sched_param param;
  int policy;
  int ret = pthread_getschedparam(m_thread_id, &policy, &param);

  if (ret == 0)
  {
    return icl_core::ThreadPriority(param.sched_priority);
  }
  else
  {
    return 0;
  }
}

bool ThreadImplPosix::setPriority(icl_core::ThreadPriority priority)
{
  // First get the current scheduling parameters.
  struct sched_param param;
  int policy;
  int ret = pthread_getschedparam(m_thread_id, &policy, &param);

  if (ret != 0)
  {
    return false;
  }

  // Change the thread priority and set it.
  param.sched_priority = priority;
  ret = pthread_setschedparam(m_thread_id, policy, &param);

  return (ret == 0);
}

bool ThreadImplPosix::start()
{
  if (pthread_create(&m_thread_id, NULL, ThreadImplPosix::runThread, this))
  {
    m_thread_id = 0;
  }

  return m_thread_id != 0;
}

icl_core::ThreadId ThreadImplPosix::threadId() const
{
  return icl_core::ThreadId(m_thread_id);
}

void * ThreadImplPosix::runThread(void *arg)
{
  ThreadImplPosix *self = static_cast<ThreadImplPosix*>(arg);
  self->m_thread->runThread();

  return NULL;
}

}
}
