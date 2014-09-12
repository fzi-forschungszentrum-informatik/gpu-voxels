// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-24
 *
 * \brief   Contains icl_core::thread::MutexImplWin32
 *
 * \b icl_core::thread::MutexImplWin32
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_MUTEX_IMPL_WIN32_H_INCLUDED
#define ICL_CORE_THREAD_MUTEX_IMPL_WIN32_H_INCLUDED

#include <Windows.h>

#include "icl_core_thread/MutexImpl.h"

namespace icl_core {
namespace thread {

class MutexImplWin32 : public MutexImpl, protected virtual icl_core::Noncopyable
{
public:
  MutexImplWin32();
  virtual ~MutexImplWin32();

  virtual bool lock();
  virtual bool lock(const TimeStamp& timeout);
  virtual bool lock(const TimeSpan& timeout);
  virtual bool tryLock();
  virtual void unlock();

private:
  HANDLE m_mutex;
};

}
}

#endif
