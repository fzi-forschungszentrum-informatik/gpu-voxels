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
 * \brief   Contains global functions for time manipulation,
 *          encapsulated into the icl_core::os namespace
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_TIME_H_INCLUDED
#define ICL_CORE_OS_TIME_H_INCLUDED

#include "icl_core/os_ns.h"

#if defined _SYSTEM_POSIX_
# include "icl_core/os_posix_time.h"
#elif defined _SYSTEM_WIN32_
# include "icl_core/os_win32_time.h"
#else
# error "No os_time implementation defined for this platform."
#endif

namespace icl_core {
namespace os {

inline void gettimeofday(struct timespec *time)
{
  ICL_CORE_OS_IMPL_NS::gettimeofday(time);
}

inline int nanosleep(const struct timespec *rqtp, struct timespec *rmtp = 0)
{
  return ICL_CORE_OS_IMPL_NS::nanosleep(rqtp, rmtp);
}

inline unsigned int sleep(unsigned int seconds)
{
  return ICL_CORE_OS_IMPL_NS::sleep(seconds);
}

inline int usleep(unsigned long useconds)
{
  return ICL_CORE_OS_IMPL_NS::usleep(useconds);
}

}
}

#endif
