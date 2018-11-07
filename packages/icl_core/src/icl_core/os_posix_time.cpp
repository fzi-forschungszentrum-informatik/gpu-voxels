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
#include "icl_core/os_posix_time.h"

#include <unistd.h>

#include "icl_core/os_lxrt.h"

#ifdef _SYSTEM_LXRT_
# include <stdlib.h>
# include <rtai_lxrt.h>
# include <rtai_posix.h>
# if defined(_SYSTEM_LXRT_33_) || defined(_SYSTEM_LXRT_35_)
#  include <mca_lxrt_extension.h>
# endif
#endif

namespace icl_core {
namespace os {
namespace hidden_posix {

void gettimeofday(struct timespec *time)
{
#ifdef _SYSTEM_LXRT_
  if (isThisLxrtTask())
  {
# if defined(_SYSTEM_LXRT_33_) || defined(_SYSTEM_LXRT_35_)
    struct timeval tv;
    mcalxrt_do_gettimeofday(&tv);
    time->tv_sec = tv.tv_sec;
    time->tv_nsec = tv.tv_usec * 1000;
# else
    RTIME real_time = rt_get_real_time_ns();
    nanos2timespec(real_time, time);
# endif
  }
  else
#endif
  {
    struct timeval tv;
    gettimeofday(&tv, 0);
    time->tv_sec = tv.tv_sec;
    time->tv_nsec = tv.tv_usec * 1000;
  }
}

int nanosleep(const struct timespec *rqtp, struct timespec *rmtp)
{
#ifdef _SYSTEM_LXRT_
  if (isThisLxrtTask())
  {
    rt_sleep(nano2count(RTIME(1000000*rqtp->tv_sec + rqtp->tv_nsec)));
    if (rmtp != NULL)
    {
      // TODO: Do this right!
      rmtp->tv_sec = 0;
      rmtp->tv_nsec = 0;
    }
    return 0;
  }
  else
#endif
  {
    return ::nanosleep(rqtp, rmtp);
  }
}

unsigned int sleep(unsigned int seconds)
{
#ifdef _SYSTEM_LXRT_
  if (isThisLxrtTask())
  {
    rt_sleep(nano2count(RTIME(1000000000*seconds)));
    return 0;
  }
  else
#endif
  {
    return ::sleep(seconds);
  }
}

int usleep(unsigned long useconds)
{
#ifdef _SYSTEM_LXRT_
  if (isThisLxrtTask())
  {
    rt_sleep(nano2count(RTIME(1000*useconds)));
    return 0;
  }
  else
#endif
  {
    return ::usleep(useconds);
  }
}

}
}
}
