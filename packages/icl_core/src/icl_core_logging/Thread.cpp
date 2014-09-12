// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-04-12
 *
 */
//----------------------------------------------------------------------
#include "Thread.h"

#include <icl_core/os_time.h>

#if defined _SYSTEM_LXRT_
# include "ThreadImplLxrt.h"
#endif

#if defined _SYSTEM_POSIX_
# include "ThreadImplPosix.h"
#elif defined _SYSTEM_WIN32_
# include "ThreadImplWin32.h"
#else
# error "No thread implementation defined for this platform."
#endif

namespace icl_core {
namespace logging {

Thread::Thread(icl_core::ThreadPriority priority)
  : m_execute(false),
    m_finished(true),
    m_joined(true),
    m_starting(false),
    m_impl(0)
{
#if defined _SYSTEM_LXRT_
  // Only create an LXRT implementation if the LXRT runtime system
  // is really available. Otherwise create an ACE or POSIX implementation,
  // depending on the system configuration.
  // Remark: This allows us to compile programs with LXRT support but run
  // them on systems on which no LXRT is installed and to disable LXRT support
  // at program startup on systems with installed LXRT!
  if (icl_core::os::isThisLxrtTask())
  {
    m_impl = new ThreadImplLxrt(this, priority);
  }
  else
  {
    m_impl = new ThreadImplPosix(this, priority);
  }

#elif defined _SYSTEM_POSIX_
  m_impl = new ThreadImplPosix(this, priority);

#elif defined _SYSTEM_WIN32_
  m_impl = new ThreadImplWin32(this, priority);

#endif
}

Thread::~Thread()
{
  if (!m_joined)
  {
    stop();
    join();
  }
  delete m_impl;
}

void Thread::join()
{
  if (running())
  {
    m_impl->join();
  }

  m_joined = true;
}

bool Thread::start()
{
  // Don't do anything if the thread is already starting or running.
  if (m_starting || running())
  {
    waitStarted();

    return running();
  }

  m_starting = true;
  m_finished = false;

  if (!m_joined)
  {
    join();
  }

  m_joined = false;

  if (!m_impl->start())
  {
    m_finished = true;
    m_starting = false;
    m_joined = true;

    return false;
  }
  else
  {
    waitStarted();

    return true;
  }
}

void Thread::runThread()
{
  m_execute = true;
  m_starting = false;
  m_finished = false;

  run();

  m_execute = false;
  m_finished = true;
}

void Thread::waitStarted() const
{
  while (m_starting)
  {
    // Sleep for 1 microsecond.
    icl_core::os::usleep(1);
  }
}

}
}
