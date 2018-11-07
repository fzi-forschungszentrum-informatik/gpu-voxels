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
 * \brief   Contains icl_core::thread::ThreadImpl for Win32 systems
 *
 * \b icl_core::thread::ThreadImpl for Win32 systems
 *
 * Wrapper class for thread implementation.
 * Uses system dependent tTheradImpl class.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_THREAD_IMPL_WIN32_H_INCLUDED
#define ICL_CORE_THREAD_THREAD_IMPL_WIN32_H_INCLUDED

#include <Process.h>
#include <Windows.h>

#include <icl_core/os_thread.h>

#include "icl_core_thread/ThreadImpl.h"

namespace icl_core {
namespace thread {

class Thread;

/*! Implements thread functionality for Win32 systems. Have a look at
 *  documentation of basic class Thread for informations about the
 *  specific functions.
 */
class ThreadImplWin32 : public ThreadImpl, protected virtual icl_core::Noncopyable
{
public:
  ThreadImplWin32(Thread *thread, const icl_core::String& description,
                  icl_core::ThreadPriority priority);

  virtual ~ThreadImplWin32();

  virtual void cancel();
  virtual icl_core::String getDescription() const { return m_description; }
  virtual bool isHardRealtime() const { return false; }
  virtual bool executesHardRealtime() const { return false; }
  virtual void join();
  virtual icl_core::ThreadPriority priority() const;
  virtual void setDescription(const icl_core::String& description) { m_description = description; }
  virtual bool setHardRealtime(bool hard_realtime) { return !hard_realtime; }
  virtual bool setPriority(icl_core::ThreadPriority priority);
  virtual bool start();
  virtual icl_core::ThreadId threadId() const;

private:
  static DWORD WINAPI runThread(void *arg);

  HANDLE m_thread_handle;
  unsigned long m_thread_id;
  Thread *m_thread;

  icl_core::ThreadPriority m_priority;
  icl_core::String m_description;
};

}
}

#endif
