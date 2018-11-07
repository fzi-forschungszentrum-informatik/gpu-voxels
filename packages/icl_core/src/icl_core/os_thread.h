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
 * \date    2008-01-28
 *
 * \brief   Contains global thread related functions,
 *          encapsulated into the icl_core::os namespace
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_THREAD_H_INCLUDED
#define ICL_CORE_OS_THREAD_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/os_ns.h"

#if defined _SYSTEM_POSIX_
# include "icl_core/os_posix_thread.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core/os_win32_thread.h"
#else
# error "No os_thread implementation defined for this platform."
#endif

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
typedef ICL_CORE_VC_DEPRECATE ICL_CORE_OS_IMPL_NS::ThreadId tThreadId ICL_CORE_GCC_DEPRECATE;
typedef ICL_CORE_VC_DEPRECATE int32_t tThreadPriority ICL_CORE_GCC_DEPRECATE;
#endif
typedef ICL_CORE_OS_IMPL_NS::ThreadId ThreadId;
typedef int32_t ThreadPriority;

namespace os {

inline ThreadId threadSelf()
{
  return ICL_CORE_OS_IMPL_NS::threadSelf();
}

inline pid_t getpid()
{
  return ICL_CORE_OS_IMPL_NS::getpid();
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

inline ThreadId ThreadSelf() ICL_CORE_GCC_DEPRECATE;
inline ICL_CORE_VC_DEPRECATE_STYLE ThreadId ThreadSelf()
{ return threadSelf(); }

#endif
/////////////////////////////////////////////////

}
}

#endif
