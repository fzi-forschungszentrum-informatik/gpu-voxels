// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \date    2010-02-08
 *
 * \brief   Contains icl_core::thread::RWLockImplWin32
 *
 * \b icl_core::thread::RWLockImplWin32
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_RWLOCK_IMPL_WIN32_H_INCLUDED
#define ICL_CORE_THREAD_RWLOCK_IMPL_WIN32_H_INCLUDED

#include <vector>
#include <Windows.h>

#include "icl_core_thread/RWLockImpl.h"
#include "icl_core_thread/Mutex.h"

namespace icl_core {
namespace thread {

class RWLockImplWin32 : public RWLockImpl, protected virtual icl_core::Noncopyable
{
public:
  RWLockImplWin32();
  virtual ~RWLockImplWin32();

  virtual bool readLock();
  virtual bool readLock(const TimeStamp& timeout);
  virtual bool readLock(const TimeSpan& timeout);
  virtual bool tryReadLock();

  virtual bool writeLock();
  virtual bool writeLock(const TimeStamp& timeout);
  virtual bool writeLock(const TimeSpan& timeout);
  virtual bool tryWriteLock();

  virtual void unlock();

private:
  bool readLock(DWORD timeout);
  bool writeLock(DWORD timeout);

  Mutex m_reader_access_lock;
  HANDLE m_reader_mutex_event;
  HANDLE m_writer_mutex;
  int m_number_of_writer;
  int m_number_of_reader;
  int m_writer_pid;
  std::vector<int> m_reader_pid;
};

}
}

#endif
