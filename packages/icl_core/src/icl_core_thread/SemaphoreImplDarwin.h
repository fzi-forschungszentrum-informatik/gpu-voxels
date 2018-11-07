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
 * \date    2009-02-02
 *
 * \brief   Contains icl_core::thread::SemaphoreImplDarwin
 *
 * \b icl_core::thread::SemaphoreImplDarwin
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SEMAPHORE_IMPL_DARWIN_H_INCLUDED
#define ICL_CORE_THREAD_SEMAPHORE_IMPL_DARWIN_H_INCLUDED

#include <mach/semaphore.h>

#include "icl_core/BaseTypes.h"
#include "icl_core_thread/SemaphoreImpl.h"

namespace icl_core {
namespace thread {

class SemaphoreImplDarwin : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplDarwin(size_t initial_value);
  virtual ~SemaphoreImplDarwin();

  virtual void post();
  virtual bool tryWait();
  virtual bool wait();
  virtual bool wait(const icl_core::TimeSpan& timeout);
  virtual bool wait(const icl_core::TimeStamp& timeout);

private:
  semaphore_t m_semaphore;
};

}
}

#endif
