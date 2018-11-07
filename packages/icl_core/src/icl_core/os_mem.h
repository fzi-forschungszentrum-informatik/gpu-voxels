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
 * \brief   Contains global functions for memory manipulation,
 *          encapsulated into the icl_core::os namespace
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_MEM_H_INCLUDED
#define ICL_CORE_OS_MEM_H_INCLUDED

#include "icl_core/os_ns.h"

#if defined _SYSTEM_POSIX_
# include "icl_core/os_posix_mem.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core/os_win32_mem.h"
#else
# error "No os_mem implementation defined for this platform."
#endif

namespace icl_core {
namespace os {

inline void *memcpy(void *dest, void *src, size_t count)
{
  return ICL_CORE_OS_IMPL_NS::memcpy(dest, src, count);
}

inline void *memset(void *dest, int c, size_t count)
{
  return ICL_CORE_OS_IMPL_NS::memset(dest, c, count);
}

}
}

#endif
