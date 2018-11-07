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
 * \date    2008-04-12
 *
 * \brief   Contains icl_core::logging::Thread
 *
 * \b icl_core::logging::Thread
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_THREAD_H_INCLUDED
#define ICL_CORE_LOGGING_THREAD_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/os_thread.h>
#include <icl_core/Noncopyable.h>
#include "icl_core_logging/ImportExport.h"

namespace icl_core {
namespace logging {

class ThreadImpl;

/*! Implements a platform independent minimal thread.
 *
 *  Remark: This class is intended to be only used in
 *  ::icl_core::logging.  Use ::icl_core::thread::Thread in your
 *  applications instead.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT Thread : protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a new thread with the \a description and \a priority.
   *  If the priority is negative and LXRT is available, the thread is
   *  scheduled in hard-realtime.
   */
  Thread(icl_core::ThreadPriority priority = 0);

  //! Deletes the thread. Stops it if it is still running.
  virtual ~Thread();

  /*! Call this from inside the thread code, if you want to check, wether
   *  the thread was demanded to stop from outside.
   *
   *  \returns \c false if the thread has been asked to stop, \c true
   *  otherwise.
   */
  bool execute() const { return m_execute; }

  /*! Wait for the thread to finish. Returns immediately if the
   *  thread is not running.
   */
  void join();

  /*! This is the function running in the thread.  This has to be
   *  reimplemented from derived classes.  If you start the thread by
   *  calling Start() this function is executed in the thread.  If you
   *  call don't want that function to be executed in the thread you
   *  could call it directly in your derived class.
   */
  virtual void run() = 0;

  //! \returns \c true if the thread is currently running.
  bool running() const { return !m_finished; }

  /*! Starts the thread.
   *
   *  \returns \c true if the thread has been started successfully, \c
   *           false if an error occured while starting the thread.
   */
  bool start();

  /*! Cooperatively stops the thread.
   *
   *  The thread's main loop must continuously check if the thread has
   *  been stopped by calling isStopped().
   */
  void stop() { waitStarted(); m_execute = false; }

private:
  /*! Executes the thread. This function is intended to be called from
   *  thread implementations.
   */
  virtual void runThread();

  //! Suspends the calling thread until thread is started.
  void waitStarted() const;

  bool m_execute;
  bool m_finished;
  bool m_joined;
  bool m_starting;

  ThreadImpl *m_impl;

  // Declare thread implementations as friends so that they have access to the
  // RunThread() function!
  friend class ThreadImplLxrt33;
  friend class ThreadImplLxrt35;
  friend class ThreadImplLxrt38;
  friend class ThreadImplPosix;
  friend class ThreadImplWin32;
};

}
}

#endif
