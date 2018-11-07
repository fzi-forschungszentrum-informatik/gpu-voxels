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
#include <sys/types.h>
#include <sys/timeb.h>

#include "icl_core/TimeSpan.h"
#include "icl_core/TimeStamp.h"
#include "icl_core/os_win32_time.h"

namespace icl_core {
namespace os {
namespace hidden_win32 {

void gettimeofday(struct timespec *time)
{
  struct _timeb tod;
  _ftime_s(&tod);
  //Care: _ftime resolves in millisec
  time->tv_sec = time_t(tod.time);
  time->tv_nsec = tod.millitm*1000000;
}

int nanosleep(const struct timespec *rqtp, struct timespec *rmtp)
{
  icl_core::TimeSpan wait_time(*rqtp);
  icl_core::TimeStamp start_time = icl_core::TimeStamp();

  DWORD sleeptime_ms = DWORD(1000*rqtp->tv_sec + rqtp->tv_nsec/1000000);
  Sleep(sleeptime_ms);
  icl_core::TimeSpan sleep_time = ::icl_core::TimeStamp() - start_time;

  // 1ms deviation is ok!
  if (sleep_time + icl_core::TimeSpan(0, 1000000) >= wait_time)
  {
    if (rmtp != 0)
    {
      rmtp->tv_sec = 0;
      rmtp->tv_nsec = 0;
    }
    return 0;
  }
  else
  {
    if (rmtp != 0)
    {
      icl_core::TimeSpan remaining_sleep_time = wait_time - sleep_time;
      rmtp->tv_sec = remaining_sleep_time.tsSec();
      rmtp->tv_nsec = remaining_sleep_time.tsNSec();
    }
    return -1;
  }
}

unsigned int sleep(unsigned int seconds)
{
  ::icl_core::TimeStamp start_time = icl_core::TimeStamp();

  ::Sleep(DWORD(1000*seconds));
  ::icl_core::TimeSpan sleep_time = ::icl_core::TimeStamp() - start_time;

  if (sleep_time.tsSec() >= seconds)
  {
    return 0;
  }
  else
  {
    return static_cast<unsigned int>(sleep_time.tsSec() - seconds);
  }
}

int usleep(unsigned long useconds)
{
  ::icl_core::TimeStamp start_time;

  ::Sleep(DWORD(useconds/1000));
  ::icl_core::TimeSpan sleep_time = ::icl_core::TimeStamp() - start_time;

  // 1ms deviation is ok!
  if (sleep_time.toUSec() + 1000 >= useconds)
  {
    return 0;
  }
  else
  {
    return -1;
  }
}


}
}
}
