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
 * \brief   Contains icl_core::logging::Semaphore
 *
 * \b icl_core::logging::Semaphore
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SEMAPHORE_H_INCLUDED
#define ICL_CORE_LOGGING_SEMAPHORE_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include "icl_core_logging/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace logging {

class SemaphoreImpl;

//! Implements a platform independent mutex.
class ICL_CORE_LOGGING_IMPORT_EXPORT Semaphore : protected virtual icl_core::Noncopyable
{
public:
  Semaphore(size_t initial_value = 1);
  virtual ~Semaphore();

  //! Increments the semaphore.
  void post();

  /*! Decrements the semaphore.  If the semaphore is unavailable this
   *  function blocks.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  bool wait();

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  //! Increments the semaphore.
  ICL_CORE_VC_DEPRECATE_STYLE void Post() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Decrements the semaphore.  If the semaphore is unavailable this
   *  function blocks.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  SemaphoreImpl *m_impl;
};

}
}

#endif
