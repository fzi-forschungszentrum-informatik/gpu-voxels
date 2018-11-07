// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-03-29
 *
 */
//----------------------------------------------------------------------
#include "icl_core/os_posix_thread.h"

#include "icl_core/os_lxrt.h"

#ifdef _SYSTEM_LXRT_
# include <stdlib.h>
# include <rtai_lxrt.h>
# include <rtai_posix.h>
#endif

namespace icl_core {
namespace os {
namespace hidden_posix {

bool operator == (const ThreadId& left, const ThreadId& right)
{
#ifdef _SYSTEM_LXRT_
  if (isThisLxrtTask())
  {
    return pthread_equal_rt(left.m_thread_id, right.m_thread_id);
  }
  else
#endif
  {
    return pthread_equal(left.m_thread_id, right.m_thread_id);
  }
}

ThreadId threadSelf()
{
#ifdef _SYSTEM_LXRT_
  if (isThisLxrtTask())
  {
    return ThreadId(pthread_self_rt());
  }
  else
#endif
  {
    return ThreadId(pthread_self());
  }
}

pid_t getpid()
{
  return ::getpid();
}

}
}
}
