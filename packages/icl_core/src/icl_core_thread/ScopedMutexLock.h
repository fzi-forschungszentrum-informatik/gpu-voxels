// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-06-08
 *
 * \brief   Contains icl_core::thread::ScopedMutexLock
 *
 * \b icl_core::thread::ScopedMutexLock
 *
 * Manages locking and unlocking of a mutex.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SCOPED_MUTEX_LOCK_H_INCLUDED
#define ICL_CORE_THREAD_SCOPED_MUTEX_LOCK_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include "icl_core_thread/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class Mutex;

/*! \brief Manages locking and unlocking of a mutes.
 *
 *  Locks or tries to lock a mutex in the constructor and unlocks it
 *  in the destructor.
 */
class ICL_CORE_THREAD_IMPORT_EXPORT ScopedMutexLock : protected virtual icl_core::Noncopyable
{
public:
  /*! Locks the \a mutex.
   *  \param mutex The mutex to use.
   *  \param force Ensure at all cost that IsLocked() is \c true.
   */
  explicit ScopedMutexLock(Mutex& mutex, bool force = true);
  /*! Unlocks the mutex.
   */
  ~ScopedMutexLock();

  /*! Check if the mutex has been successfully locked.
   */
  bool isLocked() const { return m_is_locked; }

  /*! Implicit conversion to bool.
   */
  operator bool () const { return isLocked(); }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Check if the mutex has been successfully locked.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsLocked() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  Mutex& m_mutex;
  bool m_is_locked;
};

}
}

#endif
