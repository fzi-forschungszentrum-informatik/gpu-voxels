// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \date    2010-02-08
 */
//----------------------------------------------------------------------

#include "icl_core_thread/Common.h"
#include "icl_core_thread/RWLockImplWin32.h"

namespace icl_core {
namespace thread {

RWLockImplWin32::RWLockImplWin32()
  : m_number_of_writer(0),
    m_number_of_reader(0),
    m_writer_pid(0)
{
  m_reader_mutex_event = CreateEvent(NULL, FALSE, TRUE, NULL);
  m_writer_mutex = CreateMutex(NULL, FALSE, NULL);
}

RWLockImplWin32::~RWLockImplWin32()
{
  CloseHandle(m_reader_mutex_event);
  CloseHandle(m_writer_mutex);
}

bool RWLockImplWin32::readLock()
{
  return readLock(INFINITE);
}

// ReadLock with absolute timeout
bool RWLockImplWin32::readLock(const TimeStamp& timeout)
{
  return readLock(impl::getRelativeTimeout(timeout));
}

// ReadLock with relative timeout
bool RWLockImplWin32::readLock(const TimeSpan& timeout)
{
  return readLock(DWORD(timeout.toMSec()));
}

bool RWLockImplWin32::readLock(DWORD timeout)
{
  // get the reader access
  bool ret = false;
  if (m_reader_access_lock.lock())
  {
    if (m_reader_pid.empty())
    {
      ret = WaitForSingleObject(m_reader_mutex_event, timeout) == WAIT_OBJECT_0;
    }
    if (ret || !m_reader_pid.empty())
    {
      ret = true;
      m_reader_pid.push_back(GetCurrentThreadId());
    }
    m_reader_access_lock.unlock();
  }
  return ret;
}

bool RWLockImplWin32::tryReadLock()
{
  return readLock(0);
}

bool RWLockImplWin32::writeLock()
{
  return writeLock(INFINITE);
}

// WriteLock with absolute timeout
bool RWLockImplWin32::writeLock(const TimeStamp& timeout)
{
  return writeLock(impl::getRelativeTimeout(timeout));
}

// WriteLock with relative timeout
bool RWLockImplWin32::writeLock(const TimeSpan& timeout)
{
  return writeLock(DWORD(timeout.toMSec()));
}

bool RWLockImplWin32::writeLock(DWORD timeout)
{
  bool ret = (WaitForSingleObject(m_writer_mutex, timeout) ==  WAIT_OBJECT_0)
    && (WaitForSingleObject(m_reader_mutex_event, timeout) ==  WAIT_OBJECT_0);
  if (ret)
  {
    m_writer_pid = GetCurrentThreadId();
    m_number_of_writer++;
  }
  return ret;
}

bool RWLockImplWin32::tryWriteLock()
{
  return writeLock(0);
}

void RWLockImplWin32::unlock()
{
  int thread_pid = GetCurrentThreadId();

  // writer unlock
  if (thread_pid == m_writer_pid)
  {
    ReleaseMutex(m_writer_mutex);
    m_number_of_writer--;
    if (m_number_of_writer == 0)
    {
      m_writer_pid = 0;
      SetEvent(m_reader_mutex_event);
    }
  }
  // search for reader
  else
  {
    if (m_reader_access_lock.lock(TimeSpan(30000, 0)))
    {
      std::vector<int>::iterator iter;
      for (iter = m_reader_pid.begin(); iter != m_reader_pid.end(); iter++)
      {
        if (thread_pid == *iter)
        {
          m_reader_pid.erase(iter);
          if (m_reader_pid.empty())
          {
            SetEvent(m_reader_mutex_event);
          };
          break;
        }
      }
      m_reader_access_lock.unlock();
    }
  }
}

}
}
