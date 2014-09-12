// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \date    2010-02-08
 *
 * \brief   Contains icl_core::thread::RWLockImplLxrt33
 *
 * \b icl_core::thread::RWLockImplLxrt33
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_RWLOCK_IMPL_LXRT33_H_INCLUDED
#define ICL_CORE_THREAD_RWLOCK_IMPL_LXRT33_H_INCLUDED

#include <rtai_posix.h>

#include "icl_core_thread/RWLockImpl.h"

namespace icl_core {
namespace thread {

class RWLockImplLxrt33 : public RWLockImpl, protected virtual icl_core::Noncopyable
{
public:
  RWLockImplLxrt33();
  virtual ~RWLockImplLxrt33();

  virtual bool readLock();
  virtual bool readLock(const icl_core::TimeStamp& timeout);
  virtual bool readLock(const icl_core::TimeSpan& timeout);
  virtual bool tryReadLock();

  virtual bool writeLock();
  virtual bool writeLock(const icl_core::TimeStamp& timeout);
  virtual bool writeLock(const icl_core::TimeSpan& timeout);
  virtual bool tryWriteLock();

  virtual void unlock();

private:
  pthread_rwlock_t *m_rwlock;
};

}
}

#endif
