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
 * \date    2006-06-10
 */
//----------------------------------------------------------------------

#include "icl_core/os.h"
#include "icl_core/TimeStamp.h"

#if defined _SYSTEM_LXRT_
/*# include "icl_core/tLxrt.h"*/
/* TODO: LXRT implementation */
#endif

#if defined(_IC_BUILDER_BOOST_DATE_TIME_)
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/date_time/posix_time/posix_time_types.hpp"
#endif

#ifdef _IC_BUILDER_OPENSPLICEDDS_
# include <ccpp_dds_dcps.h>
# include "icl_core/iTimeStamp.h"
#endif

namespace icl_core {

const TimeStamp TimeStamp::cZERO(0, 0);

#ifdef _IC_BUILDER_OPENSPLICEDDS_
TimeStamp::TimeStamp(const DDS::Time_t& time_stamp)
  : TimeBase(time_stamp.sec, time_stamp.nanosec)
{
}

TimeStamp::TimeStamp(const iTimeStamp& time_stamp)
  : TimeBase(time_stamp.sec, time_stamp.nsec)
{
}

TimeStamp& TimeStamp::operator = (const DDS::Time_t& time_stamp)
{
  secs = time_stamp.sec;
  nsecs = time_stamp.nanosec;
  return *this;
}

TimeStamp& TimeStamp::operator = (const iTimeStamp& time_stamp)
{
  secs = time_stamp.sec;
  nsecs = time_stamp.nsec;
  return *this;
}

TimeStamp::operator DDS::Time_t ()
{
  DDS::Time_t result;
  result.sec = secs;
  result.nanosec = nsecs;
  return result;
}

TimeStamp::operator iTimeStamp ()
{
  iTimeStamp result;
  result.sec = secs;
  result.nsec = nsecs;
  return result;
}
#endif

#if defined(_IC_BUILDER_BOOST_DATE_TIME_)
TimeStamp::TimeStamp(const boost::posix_time::ptime& ptime_stamp)
{
  boost::posix_time::ptime unix_time_base(boost::gregorian::date(1970, 1, 1));

  secs = boost::posix_time::time_period(unix_time_base, ptime_stamp).length().total_seconds();
  nsecs = int32_t(boost::posix_time::time_period(unix_time_base, ptime_stamp).
  length().total_nanoseconds());
  normalizeTime();
}

TimeStamp& TimeStamp::operator = (const boost::posix_time::ptime& ptime_stamp)
{
  boost::posix_time::ptime unix_time_base(boost::gregorian::date(1970, 1, 1));

  secs = boost::posix_time::time_period(unix_time_base, ptime_stamp).length().total_seconds();
  nsecs = int32_t(boost::posix_time::time_period(unix_time_base, ptime_stamp).
  length().total_nanoseconds());
  normalizeTime();
  return *this;
}
#endif

TimeStamp TimeStamp::now()
{
  struct timespec ts;
  os::gettimeofday(&ts);
  return TimeStamp(ts);
}

TimeStamp TimeStamp::futureMSec(uint64_t msec)
{
  TimeStamp result(msec / 1000, uint32_t((msec % 1000) * 1000000));
  result += now();
  return result;
}

TimeStamp TimeStamp::fromIso8601BasicUTC(const String& str)
{
  int32_t tm_sec = 0;
  int32_t tm_min = 0;
  int32_t tm_hour = 0;
  int32_t tm_mday = 1;
  int32_t tm_mon = 1;
  int32_t tm_year = 1970;
  if (str.size() >= 4)
  {
    tm_year = boost::lexical_cast<int32_t>(str.substr(0,4));
  }
  if (str.size() >= 6)
  {
    tm_mon = boost::lexical_cast<int32_t>(str.substr(4,2));
  }
  if (str.size() >= 8)
  {
    tm_mday = boost::lexical_cast<int32_t>(str.substr(6,2));
  }
  // Here comes the 'T', which we ignore and skip
  if (str.size() >= 11)
  {
    tm_hour = boost::lexical_cast<int32_t>(str.substr(9,2));
  }
  if (str.size() >= 13)
  {
    tm_min = boost::lexical_cast<int32_t>(str.substr(11,2));
  }
  if (str.size() >= 15)
  {
    tm_sec = boost::lexical_cast<int32_t>(str.substr(13,2));
  }
  int32_t nsecs = 0;
  // Here comes the comma, which we ignore and skip
  if (str.size() > 16)
  {
    std::string nsec_str = (str.substr(16, 9) + "000000000").substr(0, 9);
    nsecs = boost::lexical_cast<int32_t>(nsec_str);
  }

  // Jump to beginning of given year
  uint64_t days_since_epoch = 0;
  for (int32_t y = 1970; y < tm_year; ++y)
  {
    bool leap_year = (y % 400 == 0
                      || (y % 4 == 0
                          && y % 100 != 0));
    days_since_epoch += leap_year ? 366 : 365;
  }
  // Now add months in that year
  bool leap_year = (tm_year % 400 == 0
                    || (tm_year % 4 == 0
                        && tm_year % 100 != 0));
  int32_t days_per_month[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
  if (leap_year)
  {
    days_per_month[1] = 29;
  }
  for (int32_t m = 1; m < tm_mon; ++m)
  {
    days_since_epoch += days_per_month[m-1];
  }
  // Add day of month
  days_since_epoch += tm_mday-1;
  uint64_t secs = 86400 * days_since_epoch + 3600*tm_hour + 60*tm_min + tm_sec;

  return TimeStamp(secs, nsecs);
}

TimeStamp& TimeStamp::fromNow()
{
  struct timespec ts;
  os::gettimeofday(&ts);
  fromTimespec(ts);
  return *this;
}

void TimeStamp::strfTime(char* dest, size_t max_len, const char *format) const
{
  time_t time = tsSec();
  struct tm *newtime;
  newtime = gmtime(&time);
  strftime(dest, max_len, format, newtime);
}

void TimeStamp::strfLocaltime(char* dest, size_t max_len, const char *format) const
{
  time_t time = tsSec();
  struct tm *newtime = localtime(&time);
  if (newtime)
  {
    strftime(dest, max_len, format, newtime);
  }
}

String TimeStamp::formatIso8601() const
{
  char date_time_sec[20];
  strfLocaltime(date_time_sec, 20, "%Y-%m-%d %H:%M:%S");
  return String(date_time_sec);
}

String TimeStamp::formatIso8601UTC() const
{
  char date_time_sec[20];
  strfTime(date_time_sec, 20, "%Y-%m-%d %H:%M:%S");
  return String(date_time_sec);
}

String TimeStamp::formatIso8601Basic() const
{
  char date_time_sec[16], date_time_nsec[10];
  TimeStamp adjust_nsec(*this);
  while (adjust_nsec.nsecs < 0)
  {
    --adjust_nsec.secs;
    adjust_nsec.nsecs += 1000000000;
  }
  adjust_nsec.strfLocaltime(date_time_sec, 16, "%Y%m%dT%H%M%S");
  ::icl_core::os::snprintf(date_time_nsec, 10, "%09i", adjust_nsec.tsNSec());
  return String(date_time_sec) + "," + String(date_time_nsec);
}

String TimeStamp::formatIso8601BasicUTC() const
{
  char date_time_sec[16], date_time_nsec[10];
  TimeStamp adjust_nsec(*this);
  while (adjust_nsec.nsecs < 0)
  {
    --adjust_nsec.secs;
    adjust_nsec.nsecs += 1000000000;
  }
  adjust_nsec.strfTime(date_time_sec, 16, "%Y%m%dT%H%M%S");
  ::icl_core::os::snprintf(date_time_nsec, 10, "%09i", adjust_nsec.tsNSec());
  return String(date_time_sec) + "," + String(date_time_nsec);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! This static function returns a TimeStamp that contains the
   *  current System time.
   *  \deprecated Obsolete coding style.
   */
  TimeStamp TimeStamp::Now()
  {
    return now();
  }

  /*! Returns a time stamp which lies \a msec ms in the future.
   *  \deprecated Obsolete coding style.
   */
  TimeStamp TimeStamp::FutureMSec(uint64_t msec)
  {
    return futureMSec(msec);
  }

  /*! Set the timestamp to the current system time.
   *  \deprecated Obsolete coding style.
   */
  TimeStamp& TimeStamp::FromNow()
  {
    return fromNow();
  }

  void TimeStamp::Strftime(char* dest, size_t max_len, const char *format) const
  {
    strfTime(dest, max_len, format);
  }

  void TimeStamp::StrfLocaltime(char* dest, size_t max_len, const char *format) const
  {
    strfLocaltime(dest, max_len, format);
  }

  String TimeStamp::FormatIso8601() const
  {
    return formatIso8601();
  }

  uint64_t TimeStamp::TsSec() const
  {
    return tsSec();
  }
  uint32_t TimeStamp::TsUSec() const
  {
    return tsUSec();
  }
  uint32_t TimeStamp::TsNSec() const
  {
    return tsNSec();
  }

  TimeStamp TimeStamp::MaxTime()
  {
    return maxTime();
  }

#endif
/////////////////////////////////////////////////

TimeStamp& TimeStamp::operator += (const TimeStamp& other)
{
  secs += other.secs;
  nsecs += other.nsecs;
  normalizeTime();
  return *this;
}

}
