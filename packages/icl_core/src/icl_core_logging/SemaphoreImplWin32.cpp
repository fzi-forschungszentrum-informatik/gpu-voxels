// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-12-13
 */
//----------------------------------------------------------------------
#include "SemaphoreImplWin32.h"

namespace icl_core {
namespace logging {

SemaphoreImplWin32::SemaphoreImplWin32(size_t initial_value)
  : m_semaphore(NULL)
{
  m_semaphore = CreateSemaphore(NULL, LONG(initial_value), LONG_MAX, NULL);
}

SemaphoreImplWin32::~SemaphoreImplWin32()
{
  CloseHandle(m_semaphore);
}

void SemaphoreImplWin32::post()
{
  ReleaseSemaphore(m_semaphore, 1, NULL);
}

bool SemaphoreImplWin32::wait()
{
  DWORD res = WaitForSingleObject(m_semaphore, INFINITE);
  return res == WAIT_OBJECT_0;
}

}
}
