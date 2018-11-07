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
 * \brief   Contains icl_core::logging::SemaphoreImplLxrt38
 *
 * \b icl_core::logging::SemaphoreImplLxrt38
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SEMAPHORE_IMPL_LXRT38_H_INCLUDED
#define ICL_CORE_LOGGING_SEMAPHORE_IMPL_LXRT38_H_INCLUDED

#include <rtai_sem.h>

#include "icl_core/BaseTypes.h"
#include "icl_core/Noncopyable.h"
#include "SemaphoreImpl.h"

namespace icl_core {
namespace logging {

class SemaphoreImplLxrt38 : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplLxrt38(size_t initial_value);
  virtual ~SemaphoreImplLxrt38();

  virtual void post();
  virtual bool wait();

private:
  SEM *m_semaphore;
};

}
}

#endif
