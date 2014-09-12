// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-01-20
 *
 */
//----------------------------------------------------------------------

#include "Semaphore.h"

#if defined _SYSTEM_LXRT_
# include "SemaphoreImplLxrt.h"
#endif

#if defined _SYSTEM_DARWIN_
# include "SemaphoreImplDarwin.h"
#elif defined _SYSTEM_POSIX_
# include "SemaphoreImplPosix.h"
#elif defined _SYSTEM_WIN32_
# include "SemaphoreImplWin32.h"
#else
# error "No semaphore implementation defined for this platform."
#endif

namespace icl_core {
namespace logging {

Semaphore::Semaphore(size_t initial_value)
  : m_impl(0)
{
#if defined _SYSTEM_LXRT_
  // Only create an LXRT implementation if the LXRT runtime system
  // is really available. Otherwise create an ACE or POSIX implementation,
  // depending on the system configuration.
  // Remark: This allows us to compile programs with LXRT support but run
  // them on systems on which no LXRT is installed and to disable LXRT support
  // at program startup on systems with installed LXRT!
  if (icl_core::os::isLxrtAvailable())
  {
    m_impl = new SemaphoreImplLxrt(initial_value);
  }
  else
  {
    m_impl = new SemaphoreImplPosix(initial_value);
  }

#elif defined _SYSTEM_DARWIN_
  m_impl = new SemaphoreImplDarwin(initial_value);

#elif defined _SYSTEM_POSIX_
  m_impl = new SemaphoreImplPosix(initial_value);

#elif defined _SYSTEM_WIN32_
  m_impl = new SemaphoreImplWin32(initial_value);

#endif
}

Semaphore::~Semaphore()
{
  delete m_impl;
  m_impl = 0;
}

void Semaphore::post()
{
  return m_impl->post();
}

bool Semaphore::wait()
{
  return m_impl->wait();
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  //! Increments the semaphore.
  void Semaphore::Post()
  {
    post();
  }

  /*! Decrements the semaphore.  If the semaphore is unavailable this
   *  function blocks.
   *
   *  \returns \c true if the semaphore has been decremented, \c false
   *           otherwise.
   */
  bool Semaphore::Wait()
  {
    return wait();
  }

#endif
/////////////////////////////////////////////////

}
}
