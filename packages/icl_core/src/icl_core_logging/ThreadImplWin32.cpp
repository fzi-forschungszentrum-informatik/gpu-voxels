// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-04-13
 *
 */
//----------------------------------------------------------------------
#include "ThreadImplWin32.h"

#include "Thread.h"

namespace icl_core {
namespace logging {

ThreadImplWin32::ThreadImplWin32(Thread *thread, icl_core::ThreadPriority priority)
  : m_thread_handle(0),
    m_thread(thread)
{
}

ThreadImplWin32::~ThreadImplWin32()
{
  if (m_thread_handle != 0)
  {
    ::CloseHandle(m_thread_handle);
  }
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

bool ThreadImplWin32::start()
{
  m_thread_id = 0;
  m_thread_handle = ::CreateThread(NULL, 0, ThreadImplWin32::runThread, this, 0, NULL);

  return m_thread_handle != 0;
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
