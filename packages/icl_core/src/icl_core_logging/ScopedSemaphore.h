// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2010-06-16
 *
 * \brief   Contains icl_core::logging::ScopedSemaphore
 *
 * \b icl_core::logging::ScopedSemaphore
 *
 * Manages locking and unlocking of a mutex.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SCOPED_SEMAPHORE_H_INCLUDED
#define ICL_CORE_LOGGING_SCOPED_SEMAPHORE_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include "icl_core_logging/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace logging {

class Semaphore;

/*! \brief Manages locking and unlocking of a mutes.
 *
 *  Locks or tries to lock a mutex in the constructor and unlocks it
 *  in the destructor.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT ScopedSemaphore : protected virtual icl_core::Noncopyable
{
public:
  //! Decrements the \a semaphore.
  explicit ScopedSemaphore(Semaphore& semaphore);

  //! Increments the semaphore.
  ~ScopedSemaphore();

  //! Check if the semaphore has been successfully decremented.
  bool isDecremented() const { return m_is_decremented; }

  //! Implicit conversion to bool.
  operator bool () const { return isDecremented(); }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Check if the semaphore has been successfully decremented.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsDecremented() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  Semaphore& m_semaphore;
  bool m_is_decremented;
};

}
}

#endif
