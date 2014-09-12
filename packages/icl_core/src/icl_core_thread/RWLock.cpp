// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-02-08
 *
 */
//----------------------------------------------------------------------
#include "icl_core_thread/RWLock.h"

#include <icl_core/os_lxrt.h>
#include "icl_core_thread/Logging.h"

#if defined _SYSTEM_LXRT_
# include "icl_core_thread/RWLockImplLxrt.h"
#endif

#if defined _SYSTEM_POSIX_
#  include "icl_core_thread/RWLockImplPosix.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core_thread/RWLockImplWin32.h"
#else
# error "No rwlock implementation defined for this platform."
#endif

using icl_core::logging::endl;

namespace icl_core {
namespace thread {

RWLock::RWLock()
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
    LOGGING_TRACE_C(IclCoreThread, RWLock, "Initializing LXRT rwlock." << endl);
    m_impl = new RWLockImplLxrt;
  }
  else
  {
    LOGGING_TRACE_C(IclCoreThread, RWLock, "Initializing POSIX rwlock." << endl);
    m_impl = new RWLockImplPosix;
  }

#elif defined _SYSTEM_POSIX_
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Initializing POSIX rwlock." << endl);
  m_impl = new RWLockImplPosix;

#elif defined _SYSTEM_WIN32_
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Initializing WIN32 rwlock." << endl);
  m_impl = new RWLockImplWin32;

#endif
}

RWLock::~RWLock()
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Destroying rwlock." << endl);
  delete m_impl;
  m_impl = NULL;
}

bool RWLock::readLock()
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Read locking rwlock ..." << endl);
  bool result = m_impl->readLock();
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock read lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::readLock(const ::icl_core::TimeStamp& timeout)
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Read locking rwlock with absolute timeout "
                  << timeout << " ..." << endl);
  bool result = m_impl->readLock(timeout);
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock read lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::readLock(const ::icl_core::TimeSpan& timeout)
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Read locking rwlock with relative timeout "
                  << timeout << " ..." << endl);
  bool result = m_impl->readLock(timeout);
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock read lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::tryReadLock()
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Trying to read lock rwlock ..." << endl);
  bool result = m_impl->tryReadLock();
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock try read lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::writeLock()
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Write locking rwlock ..." << endl);
  bool result = m_impl->writeLock();
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock write lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::writeLock(const ::icl_core::TimeStamp& timeout)
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Write locking rwlock with absolute timeout "
                  << timeout << " ..." << endl);
  bool result = m_impl->writeLock(timeout);
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock write lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::writeLock(const ::icl_core::TimeSpan& timeout)
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Write locking rwlock with relative timeout "
                  << timeout << " ..." << endl);
  bool result = m_impl->writeLock(timeout);
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock write lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

bool RWLock::tryWriteLock()
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Trying to write lock rwlock ..." << endl);
  bool result = m_impl->tryWriteLock();
  LOGGING_TRACE_C(IclCoreThread, RWLock, "RWLock try write lock "
                  << (result ? "successful" : "failed") << "." << endl);
  return result;
}

void RWLock::unlock()
{
  LOGGING_TRACE_C(IclCoreThread, RWLock, "Unlocking rwlock." << endl);
  m_impl->unlock();
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Tries to get a shared read lock on the RWLock.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::ReadLock()
  {
    return readLock();
  }

  /*! Tries to get a shared read lock on the RWLock with an absolute
   *  \a timeout.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::ReadLock(const icl_core::TimeStamp& timeout)
  {
    return readLock(timeout);
  }

  /*! Tries to get a shared read lock on the RWLock with a relative \a
   *  timeout.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::ReadLock(const icl_core::TimeSpan& timeout)
  {
    return readLock(timeout);
  }

  /*! Tries to get a shared read lock on the RWLock without blocking.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::TryReadLock()
  {
    return tryReadLock();
  }

  /*! Tries to get a shared write lock on the RWLock.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::WriteLock()
  {
    return writeLock();
  }

  /*! Tries to get a shared write lock on the RWLock with an absolute
   *  \a timeout.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::WriteLock(const icl_core::TimeStamp& timeout)
  {
    return writeLock(timeout);
  }

  /*! Tries to get a shared write lock on the RWLock with a relative
   *  \a timeout.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::WriteLock(const icl_core::TimeSpan& timeout)
  {
    return writeLock(timeout);
  }

  /*! Tries to get a shared write lock on the RWLock without blocking.
   *  \deprecated Obsolete coding style.
   */
  bool RWLock::TryWriteLock()
  {
    return tryWriteLock();
  }

  /*! Releases a shared read or write lock on the RWLock.
   *  \deprecated Obsolete coding style.
   */
  void RWLock::Unlock()
  {
    unlock();
  }

#endif
/////////////////////////////////////////////////

}
}
