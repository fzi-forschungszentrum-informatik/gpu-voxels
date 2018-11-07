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
 * \brief   Contains global functions for string manipulation,
 *          encapsulated into the icl_core::os namespace
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_STRING_H_INCLUDED
#define ICL_CORE_OS_STRING_H_INCLUDED

#include <stdio.h>
#include <stdarg.h>

#include "icl_core/ImportExport.h"
#include "icl_core/os_ns.h"

#if defined _SYSTEM_POSIX_
# include "icl_core/os_posix_string.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core/os_win32_string.h"
#else
# error "No os_string implementation defined for this platform."
#endif

namespace icl_core {
namespace os {

ICL_CORE_IMPORT_EXPORT int snprintf(char *buffer, size_t maxlen, const char *format, ...);

inline char * stpcpy(char *dst, const char *src)
{
  return ICL_CORE_OS_IMPL_NS::stpcpy(dst, src);
}

inline char * strdup(const char *s)
{
  return ICL_CORE_OS_IMPL_NS::strdup(s);
}

ICL_CORE_IMPORT_EXPORT int vsnprintf(char *buffer, size_t maxlen, const char *format, va_list argptr);

}
}

#endif
