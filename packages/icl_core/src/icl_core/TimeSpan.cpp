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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-09-04
 *
 */
//----------------------------------------------------------------------
#include "icl_core/TimeSpan.h"

#ifdef _IC_BUILDER_OPENSPLICEDDS_
# include "icl_core/iTimeSpan.h"
#endif

namespace icl_core {

const TimeSpan TimeSpan::cZERO(0, 0);

TimeSpan::TimeSpan(int64_t sec, int32_t nsec)
  : TimeBase(sec, nsec)
{ }

TimeSpan::TimeSpan(const struct timespec& time_span)
  : TimeBase(time_span)
{ }

#ifdef _IC_BUILDER_OPENSPLICEDDS_
TimeSpan::TimeSpan(const iTimeSpan& time_span)
  : TimeBase(time_span.sec, time_span.nsec)
{ }

TimeSpan& TimeSpan::operator = (const iTimeSpan& time_span)
{
  secs = time_span.sec;
  nsecs = time_span.nsec;
  return *this;
}

TimeSpan::operator iTimeSpan ()
{
  iTimeSpan result;
  result.sec = secs;
  result.nsec = nsecs;
  return result;
}
#endif

TimeSpan& TimeSpan::fromSec(int64_t sec)
{
  secs = sec;
  nsecs = 0;
  normalizeTime();
  return *this;
}

TimeSpan& TimeSpan::fromMSec(int64_t msec)
{
  secs = msec / 1000;
  nsecs = int32_t((msec % 1000) * 1000000);
  normalizeTime();
  return *this;
}

TimeSpan& TimeSpan::fromUSec(int64_t usec)
{
  secs = usec / 1000000;
  nsecs = int32_t((usec % 1000000) * 1000);
  normalizeTime();
  return *this;
}

TimeSpan TimeSpan::createFromSec(int64_t sec)
{
  return TimeSpan().fromSec(sec);
}

TimeSpan TimeSpan::createFromMSec(int64_t msec)
{
  return TimeSpan().fromMSec(msec);
}

TimeSpan TimeSpan::createFromUSec(int64_t usec)
{
  return TimeSpan().fromUSec(usec);
}

TimeSpan& TimeSpan::operator += (const TimeSpan& span)
{
  secs += span.secs;
  nsecs += span.nsecs;
  normalizeTime();
  return *this;
}

TimeSpan& TimeSpan::operator -= (const TimeSpan& span)
{
  secs -= span.secs;
  nsecs -= span.nsecs;
  normalizeTime();
  return *this;
}

bool TimeSpan::operator != (const TimeSpan& other) const
{
  return TimeBase::operator != (other);
}

bool TimeSpan::operator == (const TimeSpan& other) const
{
  return TimeBase::operator == (other);
}

bool TimeSpan::operator < (const TimeSpan& other) const
{
  return TimeBase::operator < (other);
}

bool TimeSpan::operator > (const TimeSpan& other) const
{
  return TimeBase::operator > (other);
}

bool TimeSpan::operator <= (const TimeSpan& other) const
{
  return TimeBase::operator <= (other);
}

bool TimeSpan::operator >= (const TimeSpan& other) const
{
  return TimeBase::operator >= (other);
}

int64_t TimeSpan::tsSec() const
{
  return secs;
}

int32_t TimeSpan::tsNSec() const
{
  return nsecs;
}

int32_t TimeSpan::tsUSec() const
{
  return nsecs/1000;
}

int64_t TimeSpan::toMSec() const
{
  return secs * 1000 + nsecs / 1000000;
}

int64_t TimeSpan::toUSec() const
{
  return secs * 1000000 + nsecs / 1000;
}

int64_t TimeSpan::toNSec() const
{
  return 1000000000 * secs + nsecs;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  TimeSpan& TimeSpan::FromSec(int64_t sec)
  {
    return fromSec(sec);
  }

  TimeSpan& TimeSpan::FromMSec(int64_t msec)
  {
    return fromMSec(msec);
  }

  TimeSpan& TimeSpan::FromUSec(int64_t usec)
  {
    return fromUSec(usec);
  }

  int64_t TimeSpan::TsSec() const
  {
    return tsSec();
  }

  int32_t TimeSpan::TsNSec() const
  {
    return tsNSec();
  }

  int32_t TimeSpan::TsUSec() const
  {
    return tsUSec();
  }

  int64_t TimeSpan::ToMSec() const
  {
    return toMSec();
  }

  int64_t TimeSpan::ToUSec() const
  {
    return toUSec();
  }

  int64_t TimeSpan::ToNSec() const
  {
    return toNSec();
  }

#endif
/////////////////////////////////////////////////

std::ostream& operator << (std::ostream& stream, const TimeSpan& time_span)
{
  int64_t calc_secs = time_span.tsSec();
  int64_t calc_nsec = time_span.tsNSec();
  if (calc_secs < 0)
  {
    stream << "-";
    calc_secs = -calc_secs;
  }
  if (calc_secs > 3600)
  {
    stream << calc_secs / 3600 << "h";
    calc_secs = calc_secs % 3600;
  }
  if (calc_secs > 60)
  {
    stream << calc_secs / 60 << "m";
    calc_secs=calc_secs % 60;
  }
  if (calc_secs > 0)
  {
    stream << calc_secs << "s";
  }

  if (calc_nsec / 1000000 * 1000000 == calc_nsec)
  {
    stream << calc_nsec / 1000000 << "ms";
  }
  else if (calc_nsec / 1000 * 1000 == calc_nsec)
  {
    stream << calc_nsec << "us";
  }
  else
  {
    stream << calc_nsec << "ns";
  }

  return stream;
}

}
