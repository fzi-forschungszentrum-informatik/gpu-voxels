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
#include "ThreadImplWin32.h"

#include <icl_core/os_win32_thread.h>

#include "Thread.h"

namespace icl_core {
namespace thread {

ThreadImplWin32::ThreadImplWin32(Thread *thread, const icl_core::String& description,
                                 icl_core::ThreadPriority priority)
  : m_thread_handle(0),
    m_thread_id(0),
    m_thread(thread),
    m_priority(priority),
    m_description(description)
{
}

ThreadImplWin32::~ThreadImplWin32()
{
  if (m_thread_handle != 0)
  {
    ::CloseHandle(m_thread_handle);
  }
}

void ThreadImplWin32::cancel()
{
  ::TerminateThread(m_thread_handle, 0);
}

void ThreadImplWin32::join()
{
  DWORD result = ::WaitForSingleObject(m_thread_handle, INFINITE);
  if (result == WAIT_OBJECT_0)
  {
    m_thread_id = 0;
  }
  else
  {
    // TODO: Error handling!
  }
}

icl_core::ThreadPriority ThreadImplWin32::priority() const
{
  return m_priority;
}

bool ThreadImplWin32::setPriority(icl_core::ThreadPriority priority)
{
  // TODO: Thread priority handling.
  m_priority = priority;
  return true;
}

bool ThreadImplWin32::start()
{
  m_thread_id = 0;
  m_thread_handle = ::CreateThread(NULL, 0, ThreadImplWin32::runThread, this, 0, NULL);

  return m_thread_handle != 0;
}

icl_core::ThreadId ThreadImplWin32::threadId() const
{
  return icl_core::ThreadId(m_thread_id);
}

DWORD WINAPI ThreadImplWin32::runThread(void *arg)
{
  ThreadImplWin32 *self = static_cast<ThreadImplWin32*>(arg);

  self->m_thread_id = ::GetCurrentThreadId();
  self->m_thread->runThread();

  return 0;
}

}
}
