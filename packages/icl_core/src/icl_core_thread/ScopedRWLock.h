// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2010-09-27
 *
 * \brief   Contains icl_core::thread::ScopedRWLock
 *
 * \b icl_core::thread::ScopedRWLock
 *
 * Manages locking and unlocking of a read-write lock.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SCOPED_RWLOCK_H_INCLUDED
#define ICL_CORE_THREAD_SCOPED_RWLOCK_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include "icl_core_thread/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class RWLock;

/*! \brief Manages locking and unlocking of a read-write lock.
 *
 *  Locks or tries to lock a read-write lock in the constructor and
 *  unlocks it in the destructor.
 */
class ICL_CORE_THREAD_IMPORT_EXPORT ScopedRWLock : protected virtual icl_core::Noncopyable
{
public:
  //! The type of lock to obtain on the read-write lock.
  enum LockMode
  {
    eLM_READ_LOCK,      //!< Obtain a read lock.
    eLM_WRITE_LOCK      //!< Obtain a write lock.
  };

  /*! Locks the read-write \a lock.
   *  \param lock The read-write lock to use.
   *  \param lock_mode Use eLM_READ_LOCK for a read lock, and
   *         eLM_WRITE_LOCK for a write lock.
   *  \param force Ensure at all cost that IsLocked() is \c true.
   */
  explicit ScopedRWLock(RWLock& lock, LockMode lock_mode, bool force = true);

  /*! Unlocks the read-write lock.
   */
  ~ScopedRWLock();

  /*! Check if the read-write lock has been successfully locked.
   */
  bool isLocked() const { return m_is_locked; }

  /*!
   * Implicit conversion to bool.
   */
  operator bool () const { return isLocked(); }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Check if the read-write lock has been successfully locked.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsLocked() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  RWLock& m_lock;
  bool m_is_locked;
};

}
}

#endif
