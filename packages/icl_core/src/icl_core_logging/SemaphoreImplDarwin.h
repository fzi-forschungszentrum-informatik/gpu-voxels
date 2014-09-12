// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-02-02
 *
 * \brief   Contains icl_core::logging::SemaphoreImplDarwin
 *
 * \b icl_core::logging::SemaphoreImplDarwin
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SEMAPHORE_IMPL_DARWIN_H_INCLUDED
#define ICL_CORE_LOGGING_SEMAPHORE_IMPL_DARWIN_H_INCLUDED

#include <mach/semaphore.h>

#include <icl_core/BaseTypes.h>

#include "SemaphoreImpl.h"

namespace icl_core {
namespace logging {

class SemaphoreImplDarwin : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplDarwin(size_t initial_value);
  virtual ~SemaphoreImplDarwin();

  virtual void post();
  virtual bool wait();

private:
  semaphore_t m_semaphore;
};

}
}

#endif
