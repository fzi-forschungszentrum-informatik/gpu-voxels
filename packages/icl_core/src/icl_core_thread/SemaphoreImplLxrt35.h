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
 * \date    2010-06-30
 *
 * \brief   Contains icl_core::thread::SemaphoreImplLxrt35
 *
 * \b icl_core::thread::SemaphoreImplLxrt35
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SEMAPHORE_IMPL_LXRT35_H_INCLUDED
#define ICL_CORE_THREAD_SEMAPHORE_IMPL_LXRT35_H_INCLUDED

#include <rtai_posix.h>

#include "icl_core/BaseTypes.h"
#include "SemaphoreImpl.h"

namespace icl_core {
namespace thread {

class SemaphoreImplLxrt35 : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplLxrt35(size_t initial_value);
  virtual ~SemaphoreImplLxrt35();

  virtual void post();
  virtual bool tryWait();
  virtual bool wait();
  virtual bool wait(const icl_core::TimeSpan& timeout);
  virtual bool wait(const icl_core::TimeStamp& timeout);

private:
  sem_t *m_semaphore;
};

}
}

#endif
