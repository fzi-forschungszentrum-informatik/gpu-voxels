// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-04-13
 *
 */
//----------------------------------------------------------------------
#include "ThreadImplPosix.h"

#include "Thread.h"

namespace icl_core {
namespace logging {

ThreadImplPosix::ThreadImplPosix(Thread *thread, icl_core::ThreadPriority priority)
  : m_thread_id(0),
    m_thread(thread),
    m_priority(priority)
{ }

ThreadImplPosix::~ThreadImplPosix()
{ }

void ThreadImplPosix::join()
{
  pthread_join(m_thread_id, NULL);
}

bool ThreadImplPosix::start()
{
  if (pthread_create(&m_thread_id, NULL, ThreadImplPosix::runThread, this))
  {
    m_thread_id = 0;
  }

  return m_thread_id != 0;
}

void *ThreadImplPosix::runThread(void *arg)
{
  ThreadImplPosix *self = static_cast<ThreadImplPosix*>(arg);
  self->m_thread->runThread();

  return NULL;
}

}
}
