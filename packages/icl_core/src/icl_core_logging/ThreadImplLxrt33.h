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
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-03-02
 *
 * \brief   Contains icl_core::logging::ThreadImplLxrt33
 *
 * \b icl_core::logging::ThreadImplLxrt33
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_THREAD_IMPL_LXRT33_H_INCLUDED
#define ICL_CORE_LOGGING_THREAD_IMPL_LXRT33_H_INCLUDED

#include <rtai_posix.h>

#include <icl_core/os_thread.h>

#include "ThreadImpl.h"

#define DEFAULT_STACK_SIZE 0x4000

namespace icl_core {
namespace logging {

class Thread;

class ThreadImplLxrt33 : public ThreadImpl, protected virtual icl_core::Noncopyable
{
public:
  ThreadImplLxrt33(Thread *thread, icl_core::ThreadPriority priority);
  virtual ~ThreadImplLxrt33();

  virtual void join();
  virtual bool start();

private:
  static void *runThread(void *arg);

  pthread_t m_thread_id;
  Thread *m_thread;
  ThreadPriority m_priority;

  RT_TASK *m_rt_task;
};

}
}

#endif
