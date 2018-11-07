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
#include "Thread.h"

#include <icl_core/os_lxrt.h>
#include <icl_core/os_string.h>
#include <icl_core/os_time.h>

#include "Common.h"
#include "Logging.h"

#if defined _SYSTEM_LXRT_
# include "ThreadImplLxrt.h"
#endif

#if defined _SYSTEM_POSIX_
# include "ThreadImplPosix.h"
#elif defined _SYSTEM_WIN32_
# include "ThreadImplWin32.h"
#else
# error "No implementation specified for System dependent components"
#endif

using icl_core::logging::endl;

namespace icl_core {
namespace thread {

Thread::Thread(const icl_core::String& description, ThreadPriority priority)
  : m_execute(false),
    m_finished(true),
    m_joined(true),
    m_starting(false),
    m_thread_info(description + ", 0"),
    m_priority(priority),
    m_impl(NULL)
{
#if defined _SYSTEM_LXRT_
  // Only create an LXRT implementation if the LXRT runtime system
  // is really available. Otherwise create an ACE or POSIX implementation,
  // depending on the system configuration.
  // Remark: This allows us to compile programs with LXRT support but run
  // them on systems on which no LXRT is installed and to disable LXRT support
  // at program startup on systems with installed LXRT!
  if (icl_core::os::isLxrtAvailable())
  {
    m_impl = new ThreadImplLxrt(this, description, priority);
  }
  else
  {
    m_impl = new ThreadImplPosix(this, description, priority);
  }

#elif defined _SYSTEM_POSIX_
  m_impl = new ThreadImplPosix(this, description, priority);

#elif defined _SYSTEM_WIN32_
  m_impl = new ThreadImplWin32(this, description, priority);

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
  m_impl = NULL;
}

void Thread::cancel()
{
  LOGGING_TRACE_CO(IclCoreThread, Thread, threadInfo(), "Begin." << endl);
  waitStarted();
  if (running())
  {
    LOGGING_DEBUG_CO(IclCoreThread, Thread,  threadInfo(), "Still running." << endl);
    m_execute = false;
    m_impl->cancel();
    m_finished = true;
  }
  LOGGING_DEBUG_CO(IclCoreThread, Thread, threadInfo(), "Done." << endl);
}

bool Thread::checkHardRealtime()
{
#ifdef _SYSTEM_LXRT_
  if (threadSelf() && os::isThisHRT() && !isHardRealtime() && m_priority < 0)
  {
    return setHardRealtime(true);
  }
  else
#endif
  {
    return false;
  }
}

icl_core::String Thread::getDescription() const
{
  return m_impl->getDescription();
}

bool Thread::isHardRealtime() const
{
  return m_impl->isHardRealtime();
}

bool Thread::executesHardRealtime() const
{
  return m_impl->executesHardRealtime();
}

void Thread::join()
{
  if (running())
  {
    m_impl->join();
  }

  m_joined = true;
}

icl_core::ThreadPriority Thread::priority() const
{
  return m_impl->priority();
}

void Thread::setDescription(const icl_core::String& description)
{
  m_impl->setDescription(description);
}

bool Thread::setHardRealtime(bool hard_realtime)
{
  return m_impl->setHardRealtime(hard_realtime);
}

bool Thread::setPriority(icl_core::ThreadPriority priority)
{
  if (m_impl->setPriority(priority))
  {
    m_priority = priority;
    return true;
  }
  else
  {
    return false;
  }
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

icl_core::ThreadId Thread::threadId() const
{
  return m_impl->threadId();
}

void Thread::runThread()
{
  char buffer[1024];
#if defined(_SYSTEM_WIN32_)
  icl_core::os::snprintf(buffer, 1023, "%s, %lu", getDescription().c_str(), threadId());
#elif defined(_SYSTEM_DARWIN_)
  icl_core::os::snprintf(buffer, 1023, "%s, %p", getDescription().c_str(), threadId().m_thread_id);
#else
  icl_core::os::snprintf(buffer, 1023, "%s, %lu", getDescription().c_str(), threadId().m_thread_id);
#endif
  m_thread_info = buffer;

  LOGGING_TRACE_CO(IclCoreThread, Thread, threadInfo(), "Begin." << endl);

  m_thread_mutex.lock();
  m_execute = true;
  m_starting = false;
  m_finished = false;

  // If this is actually a periodic thread, this call makes it periodic.
  // It this is a "normal" thread, this call does nothing.
  makePeriodic();

  // Call the run loop.
  run();

  m_execute = false;
  m_thread_mutex.unlock();
  m_finished = true;

  LOGGING_TRACE_CO(IclCoreThread, Thread, threadInfo(), "Done." << endl);
}

bool Thread::wait()
{
  return wait(icl_core::TimeStamp::maxTime());
}

bool Thread::wait(const icl_core::TimeStamp& until)
{
  bool success = false;

  if (m_joined)
  {
    return true;
  }

  waitStarted();

  if (m_finished)
  {
    success = true;
  }
  else if ((until == icl_core::TimeStamp::maxTime() && m_thread_mutex.lock())
           || m_thread_mutex.lock(until))
  {
    m_thread_mutex.unlock();
  }
  else if (icl_core::TimeStamp::now() < until)
  {
    LOGGING_ERROR_CO(IclCoreThread, Thread, threadInfo(),
                     "Thread is running and we should still wait, but LockMutex() returned unexpected."
                     "The wait function will now block further until the thread is really finished."
                     "But consider that your implementation could have a failure in locking ..." << endl);

    while (icl_core::TimeStamp::now() < until && !m_finished)
    {
      os::sleep(1);
    }
  }

  if (m_finished)
  {
    success = true;
  }

  if (success)
  {
    join();
    return true;
  }
  else
  {
    LOGGING_ERROR_CO(IclCoreThread, Thread, threadInfo(), "Wait not succesful." << endl);
    return false;
  }
}

bool Thread::wait(const icl_core::TimeSpan& timeout)
{
  return wait(impl::getAbsoluteTimeout(timeout));
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Forces the thread to terminate.  If this function is called the
 *  thread has no chance to cleanup resources or its current
 *  computation.
 *
 *  Use stop() in combination with wait() or join() to cooperatively
 *  terminate a thread.
 *  \deprecated Obsolete coding style.
 */
void Thread::Cancel()
{
  cancel();
}

/*! Checks whether the RT task is in hard or soft real time mode.
 *  \deprecated Obsolete coding style.
 */
bool Thread::CheckHardRealtime()
{
  return checkHardRealtime();
}

/*! Returns the thread's description.
 *  \deprecated Obsolete coding style.
 */
icl_core::String Thread::Description() const
{
  return getDescription();
}

/*! Call this from inside the thread code if you want to check
 *  whether the thread was demanded to stop from outside.
 *  \deprecated Obsolete coding style.
 */
bool Thread::Execute() const
{
  return execute();
}

/*! Returns true if this is defined to be a hard realtime thread.
 *  \deprecated Obsolete coding style.
 */
bool Thread::IsHardRealtime() const
{
  return isHardRealtime();
}

/*! Returns true if this thread actually executes in hard realtime
 *  mode.
 *  \deprecated Obsolete coding style.
 */
bool Thread::ExecutesHardRealtime() const
{
  return executesHardRealtime();
}

/*! Wait for the thread to finish.
 *  \deprecated Obsolete coding style.
 */
void Thread::Join()
{
  join();
}

/*! Returns the thread's priority.
 *  \deprecated Obsolete coding style.
 */
icl_core::ThreadPriority Thread::Priority() const
{
  return priority();
}

/*! \returns \c true if the thread is currently running.
 *  \deprecated Obsolete coding style.
 */
bool Thread::Running() const
{
  return running();
}

/*! Set the description of the thread.
 *  \deprecated Obsolete coding style.
 */
void Thread::SetDescription(const icl_core::String& description)
{
  setDescription(description);
}

/*! Set this thread to hard or soft realtime mode.
 *  \deprecated Obsolete coding style.
 */
bool Thread::SetHardRealtime(bool hard_realtime)
{
  return setHardRealtime(hard_realtime);
}

/*! Changes the priority to the new specified value.
 *  \deprecated Obsolete coding style.
 */
bool Thread::SetPriority(icl_core::ThreadPriority priority)
{
  return setPriority(priority);
}

/*! Starts the thread.
 *  \deprecated Obsolete coding style.
 */
bool Thread::Start()
{
  return start();
}

/*! Cooperatively stops the thread.
 *  \deprecated Obsolete coding style.
 */
void Thread::Stop()
{
  stop();
}

/*! Returns the ID of this thread.
 *  \deprecated Obsolete coding style.
 */
icl_core::ThreadId Thread::ThreadId() const
{
  return threadId();
}

/*! Get the thread info, which consists of the thread description
 *  and the thread ID.
 *  \deprecated Obsolete coding style.
 */
const char *Thread::ThreadInfo() const
{
  return threadInfo();
}

/*! Check whether the calling thread is the thread running by this
 *  object.
 *  \deprecated Obsolete coding style.
 */
bool Thread::ThreadSelf() const
{
  return threadSelf();
}

/*! Waits indefinitely until the thread has finished execution.
 *  \deprecated Obsolete coding style.
 */
bool Thread::Wait()
{
  return wait();
}

/*! Waits until either the thread has finished execution or the
 *  specified absolute \a timeout elapses.
 *  \deprecated Obsolete coding style.
 */
bool Thread::Wait(const icl_core::TimeStamp& timeout)
{
  return wait(timeout);
}

/*! Waits until either the thread has finished execution or the
 *  specified relative \a timeout elapses.
 *  \deprecated Obsolete coding style.
 */
bool Thread::Wait(const icl_core::TimeSpan& timeout)
{
  return wait(timeout);
}

/*! Returns the ID of the thread in which this function is called.
 *  \deprecated Obsolete coding style.
 */
icl_core::ThreadId Thread::SelfId()
{
  return selfId();
}

#endif
/////////////////////////////////////////////////

void Thread::waitStarted() const
{
  while (m_starting)
  {
    // Sleep for 1 microsecond.
    icl_core::os::usleep(1);
  }
}

void Thread::makePeriodic()
{
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Executes the thread.
 *  \deprecated Obsolete coding style.
 */
void Thread::RunThread()
{
  runThread();
}

/*! Suspends the calling thread until thread is started.
 *  \deprecated Obsolete coding style.
 */
void Thread::WaitStarted() const
{
  waitStarted();
}

#endif
/////////////////////////////////////////////////

}
}
