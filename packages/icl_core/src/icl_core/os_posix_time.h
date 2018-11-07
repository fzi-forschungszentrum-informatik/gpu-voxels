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
 * \brief   Posix implementation of the global functions
 *          for time manipulation,
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_POSIX_TIME_H_INCLUDED
#define ICL_CORE_OS_POSIX_TIME_H_INCLUDED

#include "icl_core/ImportExport.h"

#ifdef _IC_BUILDER_HAS_TIME_H_
# include <time.h>
#endif
#ifdef _IC_BUILDER_HAS_SYS_TIME_H_
# include <sys/time.h>
#endif

namespace icl_core {
namespace os {
namespace hidden_posix {

void gettimeofday(struct timespec *time);
ICL_CORE_IMPORT_EXPORT int nanosleep(const struct timespec *rqtp, struct timespec *rmtp = 0);
ICL_CORE_IMPORT_EXPORT unsigned int sleep(unsigned int seconds);
ICL_CORE_IMPORT_EXPORT int usleep(unsigned long useconds);

}
}
}

#endif
