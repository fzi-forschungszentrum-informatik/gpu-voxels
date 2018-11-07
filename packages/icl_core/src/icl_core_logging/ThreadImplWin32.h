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
 * \date    2008-04-13
 *
 * \brief   Contains icl_core::logging::ThreadImplWin32
 *
 * \b icl_core::logging::ThreadImplWin32
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_THREAD_IMPL_WIN32_H_INCLUDED
#define ICL_CORE_LOGGING_THREAD_IMPL_WIN32_H_INCLUDED

#include <Process.h>
#include <Windows.h>

#include <icl_core/os_thread.h>

#include "ThreadImpl.h"

namespace icl_core {
namespace logging {

class Thread;

class ThreadImplWin32 : public ThreadImpl, protected virtual icl_core::Noncopyable
{
public:
  ThreadImplWin32(Thread *thread, icl_core::ThreadPriority priority);
  virtual ~ThreadImplWin32();

  virtual void join();
  virtual bool start();

private:
  static DWORD WINAPI runThread(void *arg);

  HANDLE m_thread_handle;
  unsigned long m_thread_id;
  Thread *m_thread;
};

}
}

#endif
