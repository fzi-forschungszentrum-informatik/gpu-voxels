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
 * \date    2009-12-13
 *
 * \brief   Contains icl_core::thread::SemaphoreImplWin32
 *
 * \b icl_core::thread::SemaphoreImplWin32
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SEMAPHORE_IMPL_WIN32_H_INCLUDED
#define ICL_CORE_LOGGING_SEMAPHORE_IMPL_WIN32_H_INCLUDED

#include <Windows.h>

#include "icl_core/BaseTypes.h"
#include "icl_core_logging/SemaphoreImpl.h"

namespace icl_core {
namespace logging {

class SemaphoreImplWin32 : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplWin32(size_t initial_value);
  virtual ~SemaphoreImplWin32();

  virtual void post();
  virtual bool wait();

private:
  HANDLE m_semaphore;
};

}
}

#endif
