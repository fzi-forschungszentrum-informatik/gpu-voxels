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
 * \date    2012-01-19
 *
 */
//----------------------------------------------------------------------
#include "icl_core/os_string.h"

namespace icl_core {
namespace os {

int snprintf(char *buffer, size_t maxlen, const char *format, ...)
{
  int result;
  va_list argptr;
  va_start (argptr, format);
  result = ICL_CORE_OS_IMPL_NS::vsnprintf(buffer, maxlen, format, argptr);
  va_end (argptr);
  return result;
}

int vsnprintf(char *buffer, size_t maxlen, const char *format, va_list argptr)
{
  return ICL_CORE_OS_IMPL_NS::vsnprintf(buffer, maxlen, format, argptr);
}

}}
