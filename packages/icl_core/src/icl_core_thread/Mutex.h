// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 * \brief   Contains icl_core::thread::Mutex
 *
 * \b icl_core::thread::Mutex
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_MUTEX_H_INCLUDED
#define ICL_CORE_THREAD_MUTEX_H_INCLUDED

#include "icl_core/Noncopyable.h"
#include "icl_core/TimeSpan.h"
#include "icl_core/TimeStamp.h"
#include "icl_core_thread/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class MutexImpl;

/*! Implements a platform independent mutex.
 */
class ICL_CORE_THREAD_IMPORT_EXPORT Mutex : protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a mutex.
   */
  Mutex();

  /*! Destroys a mutex.
   */
  virtual ~Mutex();

  /*! Locks a mutex.  Blocks until the mutex can be locked or a severe
   *  error occurs.  Returns \c true on success and \c false on
   *  failure.
   */
  bool lock();

  /*! Locks a mutex with an absolute \a timeout.  The function may
   *  then return with \c false, if the absolute time passes without
   *  being able to lock the mutex.
   */
  bool lock(const ::icl_core::TimeStamp& timeout);

  /*! Locks a mutex with a relative \a timeout.  The function may then
   *  return with \c false, if the relative time passes without being
   *  able to lock the mutex.
   */
  bool lock(const ::icl_core::TimeSpan& timeout);

  /*! Tries to lock a mutex without blocking.  The mutex is locked if
   *  it is available.
   *
   *  \returns \c true if the mutex has been locked, \c false
   *           otherwise.
   */
  bool tryLock();

  /*! Releases a mutex lock.
   */
  void unlock();

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Locks a mutex.  Blocks until the mutex can be locked or a severe
   *  error occurs.  Returns \c true on success and \c false on
   *  failure.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Lock() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Locks a mutex with an absolute \a timeout.  The function may
   *  then return with \c false, if the absolute time passes without
   *  being able to lock the mutex.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Lock(const icl_core::TimeStamp& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Locks a mutex with a relative \a timeout.  The function may then
   *  return with \c false, if the relative time passes without being
   *  able to lock the mutex.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Lock(const icl_core::TimeSpan& timeout)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Tries to lock a mutex without blocking.  The mutex is locked if
   *  it is available.
   *
   *  \returns \c true if the mutex has been locked, \c false
   *           otherwise.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool TryLock() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Releases a mutex lock.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Unlock() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  MutexImpl *m_impl;
};

}
}

#endif
