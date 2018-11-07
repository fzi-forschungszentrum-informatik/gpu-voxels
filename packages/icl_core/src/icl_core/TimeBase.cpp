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
 * \date    2010-04-14
 *
 */
//----------------------------------------------------------------------
#include "TimeBase.h"

#include <limits>

#include "os_lxrt.h"
#include "os_time.h"

#ifdef _SYSTEM_LXRT_33_
# include <rtai_lxrt.h>
# include <rtai_posix.h>
#endif

namespace icl_core {

TimeBase& TimeBase::operator += (const TimeBase& span)
{
  secs += span.secs;
  nsecs += span.nsecs;
  normalizeTime();
  return *this;
}

TimeBase& TimeBase::operator -= (const TimeBase& span)
{
  secs -= span.secs;
  nsecs -= span.nsecs;
  normalizeTime();
  return *this;
}

bool TimeBase::operator != (const TimeBase& other) const
{
  return (secs != other.secs) || (nsecs != other.nsecs);
}

bool TimeBase::operator == (const TimeBase& other) const
{
  return (secs == other.secs) && (nsecs == other.nsecs);
}

bool TimeBase::operator < (const TimeBase& other) const
{
  return (secs == other.secs) ? (nsecs < other.nsecs) : (secs < other.secs);
}

bool TimeBase::operator > (const TimeBase& other) const
{
  return (secs == other.secs) ? (nsecs > other.nsecs) : (secs > other.secs);
}

bool TimeBase::operator <= (const TimeBase& other) const
{
  return (secs == other.secs) ? (nsecs <= other.nsecs) : (secs < other.secs);
}

bool TimeBase::operator >= (const TimeBase& other) const
{
  return (secs == other.secs) ? (nsecs >= other.nsecs) : (secs > other.secs);
}

void TimeBase::normalizeTime()
{
  while (((secs >= 0) && (nsecs >= 1000000000)) ||
         ((secs <= 0) && (nsecs <= -1000000000)) ||
         ((secs > 0) && (nsecs < 0)) ||
         ((secs < 0) && (nsecs > 0)))
  {
    if ((secs >= 0) && (nsecs >= 1000000000))
    {
      nsecs -= 1000000000;
      ++secs;
    }
    else if ((secs <= 0) && (nsecs <= -1000000000))
    {
      nsecs += 1000000000;
      --secs;
    }
    else if ((secs > 0) && (nsecs < 0))
    {
      nsecs += 1000000000;
      --secs;
    }
    else if ((secs < 0) && (nsecs > 0))
    {
      nsecs -= 1000000000;
      ++secs;
    }
  }
}

struct timespec TimeBase::timespec() const
{
  struct timespec time;
  time.tv_sec = time_t(secs);
  time.tv_nsec = long(nsecs);
  return time;
}

struct timespec TimeBase::systemTimespec() const
{
#ifdef _SYSTEM_LXRT_33_
  if (os::isThisLxrtTask())
  {
    struct timespec time = timespec();
    struct timespec global_now;
    os::gettimeofday(&global_now);

    RTIME rtime_rtai_now = rt_get_time();
    RTIME rtime_global_now = timespec2count(&global_now);
    RTIME rtime_time = timespec2count(&time);

    count2timespec(rtime_time + rtime_rtai_now - rtime_global_now, &time);

    return time;
  }
  else
  {
#endif
    return timespec();
#ifdef _SYSTEM_LXRT_33_
  }
#endif
}

#ifdef _SYSTEM_DARWIN_
mach_timespec_t TimeBase::machTimespec() const
{
  mach_timespec_t time;
  time.tv_sec = static_cast<unsigned int>(secs);
  time.tv_nsec = static_cast<clock_res_t>(nsecs);
  return time;
}
#endif

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  void TimeBase::NormalizeTime()
  {
    normalizeTime();
  }

  int64_t TimeBase::Days() const
  {
    return days();
  }

  int64_t TimeBase::Hours() const
  {
    return hours();
  }

  int64_t TimeBase::Minutes() const
  {
    return minutes();
  }

  int64_t TimeBase::Seconds() const
  {
    return seconds();
  }

  int32_t TimeBase::MilliSeconds() const
  {
    return milliSeconds();
  }

  int32_t TimeBase::MicroSeconds() const
  {
    return microSeconds();
  }

  int32_t TimeBase::NanoSeconds() const
  {
    return nanoSeconds();
  }

  int64_t TimeBase::TbSec() const
  {
    return tbSec();
  }

  int32_t TimeBase::TbNSec() const
  {
    return tbNSec();
  }

  struct timespec TimeBase::Timespec() const
  {
    return timespec();
  }

  struct timespec TimeBase::SystemTimespec() const
  {
    return systemTimespec();
  }

#ifdef _SYSTEM_DARWIN_
  /*! Convert to \a mach_timespec_t.
   *  \deprecated Obsolete coding style.
   */
  mach_timespec_t TimeBase::MachTimespec() const
  {
    return machTimespec();
  }
#endif

#endif

TimeBase TimeBase::maxTime()
{
// TODO: Fix this in a better way!
#undef max
  return TimeBase(std::numeric_limits<int64_t>::max(), 999999999);
}

TimeBase::TimeBase(int64_t secs, int32_t nsecs)
  : secs(secs),
    nsecs(nsecs)
{
  normalizeTime();
}

TimeBase::TimeBase(const struct timespec& time)
  : secs(time.tv_sec),
    nsecs(time.tv_nsec)
{
  normalizeTime();
}

void TimeBase::fromTimespec(const struct timespec& time)
{
  secs = time.tv_sec;
  nsecs = time.tv_nsec;
  normalizeTime();
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  TimeBase TimeBase::MaxTime()
  {
    return maxTime();
  }

  void TimeBase::FromTimespec(const struct timespec& time)
  {
    fromTimespec(time);
  }

#endif
/////////////////////////////////////////////////

}
