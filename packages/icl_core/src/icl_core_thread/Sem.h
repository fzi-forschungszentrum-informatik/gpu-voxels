// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-01-20
 *
 * \brief   Contains icl_core::thread::Semaphore
 *
 * \b icl_core::thread::Semaphore
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SEM_H_INCLUDED
#define ICL_CORE_THREAD_SEM_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include <icl_core/TimeSpan.h>
#include <icl_core/TimeStamp.h>

#include "icl_core_thread/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class SemaphoreImpl;

/*! Implements a platform independent mutex.
 */
class ICL_CORE_THREAD_IMPORT_EXPORT Semaphore : protected virtual icl_core::Noncopyable
{
public:
  Semaphore(size_t initial_value);
  virtual ~Semaphore();

  /*! Increments the semaphore.
   */
  void post();

  /*! Tries to decrement the semaphore.  Does not block if the
   *  semaphore is unavailable.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  bool tryWait();

  /*! Decrements the semaphore.  If the semaphore is unavailable this
   *  function blocks.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  bool wait();

  /*! Decrements the semaphore.  If the semaphore is unavailable this
   *  function blocks until the relative \a timeout has elapsed.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  bool wait(const icl_core::TimeSpan& timeout);

  /*! Decrements the semaphore.  If the semaphore is unavailable this
   *  function blocks until the absolute \a timeout has elapsed.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  bool wait(const icl_core::TimeStamp& timeout);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Increments the semaphore.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Post() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to decrement the semaphore.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool TryWait() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Decrements the semaphore.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Decrements the semaphore.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait(const icl_core::TimeSpan& timeout) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Decrements the semaphore.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Wait(const icl_core::TimeStamp& timeout) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  SemaphoreImpl *m_impl;
};

}
}

#endif
