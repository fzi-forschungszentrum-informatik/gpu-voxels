// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2010-06-16
 *
 */
//----------------------------------------------------------------------

#include "icl_core_logging/ScopedSemaphore.h"
#include "icl_core_logging/Semaphore.h"

namespace icl_core {
namespace logging {

ScopedSemaphore::ScopedSemaphore(Semaphore& semaphore)
  : m_semaphore(semaphore),
    m_is_decremented(false)
{
  if (m_semaphore.wait())
  {
    m_is_decremented = true;
  }
}

ScopedSemaphore::~ScopedSemaphore()
{
  if (isDecremented())
  {
    m_semaphore.post();
  }
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Check if the semaphore has been successfully decremented.
 *  \deprecated Obsolete coding style.
 */
bool ScopedSemaphore::IsDecremented() const
{
  return isDecremented();
}

#endif
/////////////////////////////////////////////////

}
}
