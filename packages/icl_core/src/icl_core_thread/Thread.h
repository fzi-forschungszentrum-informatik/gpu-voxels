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
 *
 * \brief   Contains icl_core::thread::Thread
 *
 * \b icl_core::thread::Thread
 *
 * Wrapper class for thread implementation.
 * Uses system dependent ThreadImpl class.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_THREAD_H_INCLUDED
#define ICL_CORE_THREAD_THREAD_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include <icl_core/TimeSpan.h>
#include <icl_core/TimeStamp.h>
#include <icl_core/os_thread.h>

#include "icl_core_thread/ImportExport.h"
#include "icl_core_thread/Mutex.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class ThreadImpl;

/*! Only for base functionality.  You may use functions of this class,
 *  but never create one.  Use template class TThreadContainer
 *  instead.
 *
 *  For developers: The ThreadImpl class must implement the following
 *  functions:
 *  \li bool StartImpl(): return true if a thread was succesfully
 *      created
 *  \li void CancelImpl(): terminates the thread
 *  \li bool ThreadSelfImpl(): return true if calling thread is this
 *      thread
 *  \li ThreadId ThreadIdImpl(): returns the thread ID of this thread
 *  \li static ThreadId SelfIdImpl(): returns the thread id of the
 *      calling thread
 *  \li bool IsHardRealtimeImpl(): returns true if the thread is still
 *      in HRT mode
 *  \li bool IsHardRealtimeThreadImpl(): returns true is the thread is
 *      supposed to be HRT
 *  \li void SleepImpl(TimeSpan wait): suspend calling thread for the
 *      \a wait timespan (internally used)
 *  \li int JoinImpl(): waits until the thread has finished.
 *  \li int PriorityImpl(): returns the priority of this thread.
 *  \li bool SetRealtimeModeImpl(bool hard): sets this thread to be a
 *      HRT thread. returns false, if the system does not support HRT
 */
class ICL_CORE_THREAD_IMPORT_EXPORT Thread : protected virtual icl_core::Noncopyable
{
public:
  /*! Constructor.  If the priority is negative and LXRT is available,
   *  the thread is processed in hard realtime.
   */
  Thread(const icl_core::String& description, icl_core::ThreadPriority priority = 0);

  /*! Deletes the thread.  Stops it if it is still running.
   */
  virtual ~Thread();

  /*! Forces the thread to terminate.  If this function is called the
   *  thread has no chance to cleanup resources or its current
   *  computation.
   *
   *  Use stop() in combination with wait() or join() to cooperatively
   *  terminate a thread.
   */
  void cancel();

  /*! Checks whether the RT task is in hard or soft real time mode.
   *  Returns false, if the thread is in hard real time mode or is no
   *  hard real time thread at all.
   *
   *  If this function is called from inside the thread, it will be
   *  reset into hard real time mode if necessary.
   */
  bool checkHardRealtime();

  /*! Returns the thread's description.
   */
  icl_core::String getDescription() const;

  /*! Call this from inside the thread code if you want to check
   *  whether the thread was demanded to stop from outside.
   *
   *  \returns \a false if the thread has been asked to stop, \a
   *  true otherwise.
   */
  bool execute() const { return m_execute; }

  /*! Returns true if this is defined to be a hard realtime thread.
   */
  bool isHardRealtime() const;

  /*! Returns true if this thread actually executes in hard realtime
   *  mode.
   */
  bool executesHardRealtime() const;

  /*! Wait for the thread to finish. Returns immediately if the thread
   *  is not running.
   */
  void join();

  /*! Returns the thread's priority.
   */
  icl_core::ThreadPriority priority() const;

  /*! This function is called from the Stop() function.  Subclasses
   *  can implement this function, if they need to do special
   *  processing when the thread is stopped.
   */
  virtual void onStop() { }

  /*! This is the function running in the thread.  This has to be
   *  reimplemented from derived classes.  If you start the thread by
   *  calling Start() this function is executed in the thread.  If you
   *  call don't want that function to be executed in the thread you
   *  could call it directly in your derived class.
   */
  virtual void run() = 0;

  /*! \returns \c true if the thread is currently running.
   */
  bool running() const { return !m_finished; }

  /*! Set the description of the thread.
   *
   *  Depending on the platform this description is either only kept
   *  within the icl_core implementation or - if supported - also
   *  handed over to the system.
   */
  void setDescription(const icl_core::String& description);

  /*! Set this thread to hard or soft realtime mode.
   *
   *  \returns \c true if the thread could be set to the desired
   *           realtime mode, \c false otherwise.
   */
  bool setHardRealtime(bool hard_realtime = true);

  /*! Changes the priority to the new specified value.  Returns \c
   *  false if not successful.  In the constructor of Thread the
   *  Priority is set to a sensible default value that depends on the
   *  underlying operation system.
   */
  bool setPriority(icl_core::ThreadPriority priority);

  /*! Starts the thread.
   *
   *  \returns \c true if the thread has been started successfully, \c
   *           false if an error occured while starting the thread.
   */
  bool start();

  /*! Cooperatively stops the thread.
   *
   *  The thread's main loop must continuously check if the thread has
   *  been stopped by calling IsStopped().
   */
  void stop() { waitStarted(); m_execute = false; }

  /*! Returns the ID of this thread.
   */
  icl_core::ThreadId threadId() const;

  /*! Get the thread info, which consists of the thread description
   *  and the thread ID.  This is mainly used for internal logging
   *  messages.
   */
  const char *threadInfo() const { return m_thread_info.c_str(); }

  /*! Check whether the calling thread is the thread running by this
   *  object.
   */
  bool threadSelf() const
  {
    waitStarted();
    return icl_core::os::threadSelf() == threadId();
  }

  /*! Waits indefinitely until the thread has finished execution.
   *
   *  \returns Always returns \c true.
   */
  bool wait();

  /*! Waits until either the thread has finished execution or the
   *  specified absolute \a timeout elapses.
   *
   *  \returns \c true if the thread has finished or \c false if the
   *           timeout has elapsed.
   */
  bool wait(const icl_core::TimeStamp& timeout);

  /*! Waits until either the thread has finished execution or the
   *  specified relative \a timeout elapses.
   *
   *  \returns \c true if the thread has finished or \c false if the
   *           timeout has elapsed.
   */
  bool wait(const icl_core::TimeSpan& timeout);

  /*! Returns the ID of the thread in which this function is called.
   */
  static icl_core::ThreadId selfId() { return icl_core::os::threadSelf(); }

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
  ICL_CORE_VC_DEPRECATE_STYLE void Cancel() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks whether the RT task is in hard or soft real time mode.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool CheckHardRealtime() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the thread's description.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Description() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Call this from inside the thread code if you want to check
   *  whether the thread was demanded to stop from outside.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Execute() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns true if this is defined to be a hard realtime thread.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsHardRealtime() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns true if this thread actually executes in hard realtime
   *  mode.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ExecutesHardRealtime() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Wait for the thread to finish.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Join() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the thread's priority.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::ThreadPriority Priority() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! \returns \c true if the thread is currently running.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Running() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Set the description of the thread.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetDescription(const icl_core::String& description)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Set this thread to hard or soft realtime mode.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool SetHardRealtime(bool hard_realtime = true)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Changes the priority to the new specified value.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool SetPriority(icl_core::ThreadPriority priority)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Starts the thread.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Start() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Cooperatively stops the thread.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Stop() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the ID of this thread.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::ThreadId ThreadId() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the thread info, which consists of the thread description
   *  and the thread ID.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const char *ThreadInfo() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check whether the calling thread is the thread running by this
   *  object.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ThreadSelf() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Waits indefinitely until the thread has finished execution.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Waits until either the thread has finished execution or the
   *  specified absolute \a timeout elapses.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait(const icl_core::TimeStamp& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Waits until either the thread has finished execution or the
   *  specified relative \a timeout elapses.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait(const icl_core::TimeSpan& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the ID of the thread in which this function is called.
   *  \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE icl_core::ThreadId SelfId()
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  /*! Makes the thread periodic.
   *
   *  This function is overridden in PeriodicThread.
   */
  virtual void makePeriodic();

  /*! Executes the thread.  This function is intended to be called
   *  from thread implementations.
   */
  void runThread();

  /*! Suspends the calling thread until thread is started.
   */
  void waitStarted() const;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Executes the thread.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RunThread() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Suspends the calling thread until thread is started.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void WaitStarted() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

  bool m_execute;
  bool m_finished;
  bool m_joined;
  bool m_starting;

  icl_core::String m_thread_info;
  icl_core::ThreadPriority m_priority;

  Mutex m_thread_mutex;

  ThreadImpl *m_impl;

  // Declare thread implementations as friends so that they have
  // access to the runThread() function!
  friend class ThreadImplLxrt33;
  friend class ThreadImplLxrt35;
  friend class ThreadImplLxrt38;
  friend class ThreadImplPosix;
  friend class ThreadImplWin32;
};

}
}

#endif
