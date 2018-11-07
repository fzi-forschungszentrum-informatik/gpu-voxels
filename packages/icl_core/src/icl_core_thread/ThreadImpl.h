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
 * \date    2008-04-14
 *
 * \brief   Contains icl_core::thread::ThreadImpl
 *
 * \b icl_core::thread::ThreadImpl
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_THREAD_IMPL_H_INCLUDED
#define ICL_CORE_THREAD_THREAD_IMPL_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/os_thread.h>
#include <icl_core/Noncopyable.h>
#include <icl_core/TimeSpan.h>
#include <icl_core/TimeStamp.h>

namespace icl_core {
namespace thread {

class ThreadImpl : protected virtual icl_core::Noncopyable
{
public:
  virtual ~ThreadImpl() {}

  virtual void cancel() = 0;
  virtual icl_core::String getDescription() const = 0;
  virtual bool isHardRealtime() const = 0;
  virtual bool executesHardRealtime() const = 0;
  virtual void join() = 0;
  virtual icl_core::ThreadPriority priority() const = 0;
  virtual void setDescription(const icl_core::String& description) = 0;
  virtual bool setHardRealtime(bool hard_realtime) = 0;
  virtual bool setPriority(icl_core::ThreadPriority priority) = 0;
  virtual bool start() = 0;
  virtual icl_core::ThreadId threadId() const = 0;
};

}
}

#endif
