// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-01-20
 *
 * \brief   Contains icl_core::thread::SemaphoreImplPosix
 *
 * \b icl_core::thread::SemaphoreImplPosix
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SEMAPHORE_IMPL_POSIX_H_INCLUDED
#define ICL_CORE_THREAD_SEMAPHORE_IMPL_POSIX_H_INCLUDED

#include <semaphore.h>

#include <icl_core/BaseTypes.h>

#include "icl_core_thread/SemaphoreImpl.h"

namespace icl_core {
namespace thread {

class SemaphoreImplPosix : public SemaphoreImpl, protected virtual icl_core::Noncopyable
{
public:
  SemaphoreImplPosix(size_t initial_value);
  virtual ~SemaphoreImplPosix();

  virtual void post();
  virtual bool tryWait();
  virtual bool wait();
  virtual bool wait(const icl_core::TimeSpan& timeout);
  virtual bool wait(const icl_core::TimeStamp& timeout);

private:
  sem_t *m_semaphore;
};

}
}

#endif
