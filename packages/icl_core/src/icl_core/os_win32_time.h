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
 * \brief   Win32 implementation of the global functions
 *          for time manipulation,
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_WIN32_TIME_H_INCLUDED
#define ICL_CORE_OS_WIN32_TIME_H_INCLUDED

#include <time.h>
#include <Windows.h>

#include "icl_core/ImportExport.h"

struct ICL_CORE_IMPORT_EXPORT timespec
{
  /// Seconds
  time_t tv_sec;
  /// Nanoseconds
  long tv_nsec;
};

namespace icl_core {
namespace os {
namespace hidden_win32 {

ICL_CORE_IMPORT_EXPORT void gettimeofday(struct timespec *time);
ICL_CORE_IMPORT_EXPORT int nanosleep(const struct timespec *rqtp, struct timespec *rmtp = 0);
ICL_CORE_IMPORT_EXPORT unsigned int sleep(unsigned int seconds);
ICL_CORE_IMPORT_EXPORT int usleep(unsigned long useconds);

}
}
}

#endif
