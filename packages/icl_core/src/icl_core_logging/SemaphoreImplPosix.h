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
 * \date    2009-01-20
 *
 * \brief   Contains icl_core::thread::SemaphoreImplPosix
 *
 * \b icl_core::thread::SemaphoreImplPosix
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SEMAPHORE_IMPL_POSIX_H_INCLUDED
#define ICL_CORE_LOGGING_SEMAPHORE_IMPL_POSIX_H_INCLUDED

#include <semaphore.h>

#include "icl_core/BaseTypes.h"
#include "icl_core_logging/SemaphoreImpl.h"

namespace icl_core {
namespace logging {

class SemaphoreImplPosix : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplPosix(size_t initial_value);
  virtual ~SemaphoreImplPosix();

  virtual void post();
  virtual bool wait();

private:
  sem_t *m_semaphore;
};

}
}

#endif
