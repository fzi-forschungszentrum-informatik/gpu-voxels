// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2009-06-30
 *
 * \brief   Contains icl_core::thread::SemaphoreImplWin32
 *
 * \b icl_core::thread::SemaphoreImplWin32
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SEMAPHORE_IMPL_WIN32_H_INCLUDED
#define ICL_CORE_THREAD_SEMAPHORE_IMPL_WIN32_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core_thread/SemaphoreImpl.h"

namespace icl_core {
namespace thread {

class SemaphoreImplWin32 : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplWin32(size_t initial_value);
  virtual ~SemaphoreImplWin32();

  virtual void post();
  virtual bool tryWait();
  virtual bool wait();
  virtual bool wait(const icl_core::TimeSpan& timeout);
  virtual bool wait(const icl_core::TimeStamp& timeout);

private:
  void *m_semaphore;
};

}
}

#endif
