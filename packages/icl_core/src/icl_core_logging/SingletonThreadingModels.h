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
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2009-06-16
 *
 * \brief   Contains a semaphore-based threading model for icl_core::tSingleton.
 *
 * \b icl_core::tSTMMultiThreadedWithSemaphore is a multi-threaded, thread-safe
 * threading model for icl_core::tSingleton.  It performs locking using
 * icl_core::logging::tScopedSemaphore.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SINGLETON_THREADING_MODELS_H_INCLUDED
#define ICL_CORE_LOGGING_SINGLETON_THREADING_MODELS_H_INCLUDED

#include "icl_core_logging/tSemaphore.h"
#include "icl_core_logging/tScopedSemaphore.h"

namespace icl_core {
namespace logging {

//! Semaphore-based thread-safe singleton threading model.
template
<class T>
class STMMultiThreadedWithSemaphore
{
public:
  //! Memory barrier for synchronization.
  static inline void memoryBarrier()
  {
#if defined(_SYSTEM_WIN32_)
    ::MemoryBarrier();
#elif defined(__GNUC__)
    __asm__ __volatile__ ("" ::: "memory");
#else
#error "No memory barrier implementation is available for your system."
#endif
  }

  //! Use ScopedSemaphore as the lock guard.
  typedef ::icl_core::logging::ScopedSemaphore Guard;

  //! Use Semaphore as the actual lock.
  typedef ::icl_core::logging::Semaphore Lock;
};

}
}

#endif
