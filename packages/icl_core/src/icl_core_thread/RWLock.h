// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-02-08
 *
 * \brief   Contains icl_core::thread::RWLock
 *
 * \b icl_core::thread::RWLock
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_RWLOCK_H_INCLUDED
#define ICL_CORE_THREAD_RWLOCK_H_INCLUDED

#include "icl_core/Noncopyable.h"
#include "icl_core/TimeSpan.h"
#include "icl_core/TimeStamp.h"
#include "icl_core_thread/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class RWLockImpl;

/*! Implements a platform independent read/write lock.
 */
class ICL_CORE_THREAD_IMPORT_EXPORT RWLock : protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a read/write lock.
   */
  RWLock();

  /*! Destroys a read/write lock.
   */
  virtual ~RWLock();

  /*! Tries to get a shared read lock on the RWLock.
   *  \returns \c true if locking succeeds, or \c false on a timeout
   *           or every other failure.
   */
  bool readLock();

  /*! Tries to get a shared read lock on the RWLock with an absolute
   *  \a timeout.  The function may then return with \c false if the
   *  absolute time passes without being able to lock the RWLock.
   */
  bool readLock(const ::icl_core::TimeStamp& timeout);

  /*! Tries to get a shared read lock on the RWLock with a relative \a
   *  timeout.  The function may then return with \c false if the
   *  relative time passes without being able to lock the RWLock.
   */
  bool readLock(const ::icl_core::TimeSpan& timeout);

  /*! Tries to get a shared read lock on the RWLock without blocking.
   *  The mutex is locked if it is available.
   *
   *  \returns \c true if the RWLock has been locked, \c false
   *           otherwise.
   */
  bool tryReadLock();

  /*! Tries to get a shared write lock on the RWLock.
   *  \returns \c true if locking succeeds, or \c false on a timeout
   *           or every other failure.
   */
  bool writeLock();

  /*! Tries to get a shared write lock on the RWLock with an absolute
   *  \a timeout. The function may then return with \c false if the
   *  absolute time passes without being able to lock the RWLock.
   */
  bool writeLock(const ::icl_core::TimeStamp& timeout);

  /*! Tries to get a shared write lock on the RWLock with a relative
   *  \a timeout.  The function may then return with \c false if the
   *  relative time passes without being able to lock the RWLock.
   */
  bool writeLock(const ::icl_core::TimeSpan& timeout);

  /*! Tries to get a shared write lock on the RWLock without blocking.
   *  The RWLock is locked if it is available.
   *
   *  \returns \c true if the RWLock has been locked, \c false
   *           otherwise.
   */
  bool tryWriteLock();

  /*! Releases a shared read or write lock on the RWLock.
   */
  void unlock();

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Tries to get a shared read lock on the RWLock.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ReadLock() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared read lock on the RWLock with an absolute
   *  \a timeout.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ReadLock(const icl_core::TimeStamp& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared read lock on the RWLock with a relative \a
   *  timeout.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool ReadLock(const icl_core::TimeSpan& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared read lock on the RWLock without blocking.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool TryReadLock() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared write lock on the RWLock.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool WriteLock() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared write lock on the RWLock with an absolute
   *  \a timeout.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool WriteLock(const icl_core::TimeStamp& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared write lock on the RWLock with a relative
   *  \a timeout.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool WriteLock(const icl_core::TimeSpan& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to get a shared write lock on the RWLock without blocking.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool TryWriteLock() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Releases a shared read or write lock on the RWLock.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Unlock() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

private:
  RWLockImpl *m_impl;
};

}
}

#endif
