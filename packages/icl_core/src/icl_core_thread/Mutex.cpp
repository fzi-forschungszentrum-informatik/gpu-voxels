// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-01-04
 *
 */
//----------------------------------------------------------------------
#include "icl_core_thread/Mutex.h"


#include <icl_core/os_lxrt.h>

#undef ICL_CORE_LOCAL_LOGGING
//#define ICL_CORE_LOCAL_LOGGING
#include "icl_core_thread/Logging.h"

#define LOCAL_PRINTF(args)
//#define LOCAL_PRINTF PRINTF

#if defined _SYSTEM_LXRT_
# include "icl_core_thread/MutexImplLxrt.h"
#endif

#if defined _SYSTEM_POSIX_
#  include "icl_core_thread/MutexImplPosix.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core_thread/MutexImplWin32.h"
#else
# error "No mutex implementation defined for this platform."
#endif

using icl_core::logging::endl;

namespace icl_core {
namespace thread {

Mutex::Mutex()
  : m_impl(NULL)
{
#if defined _SYSTEM_LXRT_
  // Only create an LXRT implementation if the LXRT runtime system is
  // really available. Otherwise create an ACE or POSIX
  // implementation, depending on the system configuration.
  // Remark: This allows us to compile programs with LXRT support but
  // run them on systems on which no LXRT is installed and to disable
  // LXRT support at program startup on systems with installed LXRT!
  if (icl_core::os::isLxrtAvailable())
  {
    LOCAL_PRINTF("Initializing LXRT mutex.\n");
    m_impl = new MutexImplLxrt;
  }
  else
  {
    LOCAL_PRINTF("Initializing POSIX mutex.\n");
    m_impl = new MutexImplPosix;
  }

#elif defined _SYSTEM_POSIX_
  LOCAL_PRINTF("Initializing POSIX mutex.\n");
  m_impl = new MutexImplPosix;

#elif defined _SYSTEM_WIN32_
  LOCAL_PRINTF("Initializing WIN32 mutex.\n");
  m_impl = new MutexImplWin32;

#endif
}

Mutex::~Mutex()
{
  LOCAL_PRINTF("Destroying mutex.\n");
  delete m_impl;
  m_impl = NULL;
}

bool Mutex::lock()
{
  LLOGGING_TRACE_C(Thread, Mutex, "Locking mutex ..." << endl);
  bool result = m_impl->lock();
  LLOGGING_TRACE_C(Thread, Mutex, "Mutex lock " << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool Mutex::lock(const ::icl_core::TimeStamp& timeout)
{
  LLOGGING_TRACE_C(Thread, Mutex, "Locking mutex with absolute timeout " << timeout << " ..." << endl);
  bool result = m_impl->lock(timeout);
  LLOGGING_TRACE_C(Thread, Mutex, "Mutex lock " << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool Mutex::lock(const ::icl_core::TimeSpan& timeout)
{
  LLOGGING_TRACE_C(Thread, Mutex, "Locking mutex with relative timeout " << timeout << " ..." << endl);
  bool result = m_impl->lock(timeout);
  LLOGGING_TRACE_C(Thread, Mutex, "Mutex lock " << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool Mutex::tryLock()
{
  LLOGGING_TRACE_C(Thread, Mutex, "Trying to lock mutex ..." << endl);
  bool result = m_impl->tryLock();
  LLOGGING_TRACE_C(Thread, Mutex, "Mutex try-lock " << (result ? "successful" : "failed") << "." << endl);
  return result;
}

void Mutex::unlock()
{
  LLOGGING_TRACE_C(Thread, Mutex, "Unlocking mutex." << endl);
  m_impl->unlock();
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Locks a mutex.  Blocks until the mutex can be locked or a severe
 *  error occurs.  Returns \c true on success and \c false on
 *  failure.
 */
bool Mutex::Lock()
{
  return lock();
}

/*! Locks a mutex with an absolute \a timeout.  The function may
 *  then return with \c false, if the absolute time passes without
 *  being able to lock the mutex.
 */
bool Mutex::Lock(const icl_core::TimeStamp& timeout)
{
  return lock(timeout);
}

/*! Locks a mutex with a relative \a timeout.  The function may then
 *  return with \c false, if the relative time passes without being
 *  able to lock the mutex.
 */
bool Mutex::Lock(const icl_core::TimeSpan& timeout)
{
  return lock(timeout);
}

/*! Tries to lock a mutex without blocking.  The mutex is locked if
 *  it is available.
 *
 *  \returns \c true if the mutex has been locked, \c false
 *           otherwise.
 */
bool Mutex::TryLock()
{
  return tryLock();
}

/*! Releases a mutex lock.
 */
void Mutex::Unlock()
{
  unlock();
}

#endif
/////////////////////////////////////////////////

}
}
