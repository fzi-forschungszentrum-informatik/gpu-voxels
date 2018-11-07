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
 * \brief   Contains TimeStamp
 *
 * \b tTime
 *
 * Contains the definitions of a generic interface  to the system time
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_TIME_STAMP_H_INCLUDED
#define ICL_CORE_TIME_STAMP_H_INCLUDED

#include "icl_core/ImportExport.h"
#include "icl_core/TimeSpan.h"

#if defined(_IC_BUILDER_BOOST_DATE_TIME_)
// Forward declaration
namespace boost {
namespace posix_time {
class ptime;
}
}
#endif

#ifdef _IC_BUILDER_OPENSPLICEDDS_
namespace DDS {
struct Time_t;
}
#endif

namespace icl_core {

#ifdef _IC_BUILDER_OPENSPLICEDDS_
struct iTimeStamp;
#endif

//! Represents absolute times.
/*! Use this class whenever you want to deal with times, as it
 *  provides a number of useful operators and functions.
 */
class ICL_CORE_IMPORT_EXPORT TimeStamp : protected TimeBase
{
public:
  //! Standard constructor, creates a null time.
  TimeStamp()
    : TimeBase()
  { }

  //! Constructor, takes a timeval for creation.
  //TimeStamp(timeval time) { secs = time.tv_sec; nsecs = time.tv_usec * 1000; }

  //! Constructor that gets a time in seconds plus nanoseconds.
  TimeStamp(uint64_t sec, uint32_t nsec)
    : TimeBase(sec, nsec)
  { }

  TimeStamp(const struct timespec& ts)
    : TimeBase(ts)
  { }

  explicit TimeStamp(time_t timestamp)
    : TimeBase(uint64_t(timestamp), 0)
  { }

#ifdef _IC_BUILDER_OPENSPLICEDDS_
  //! Create a time stamp from a DDS time stamp.
  TimeStamp(const DDS::Time_t& time_stamp);

  //! Create a time stamp from its corresponding IDL datatype.
  TimeStamp(const iTimeStamp& time_stamp);

  //! Assign from a DDS time stamp.
  TimeStamp& operator = (const DDS::Time_t& time_stamp);

  //! Assign from an IDL time stamp.
  TimeStamp& operator = (const iTimeStamp& time_stamp);

  //! Implicit conversion to a DDS time stamp.
  operator DDS::Time_t ();

  //! Implicit conversion to an IDL time stamp.
  operator iTimeStamp ();
#endif

#if defined(_IC_BUILDER_BOOST_DATE_TIME_)
  //! Create a time stamp from a boost posix_time stamp.
  TimeStamp(const boost::posix_time::ptime& ptime_stamp);

  //! Assign from a boost posix_time stamp.
  TimeStamp& operator = (const boost::posix_time::ptime& ptime_stamp);
#endif

  /*! This static function returns a TimeStamp that contains the
   *  current System time (as UTC).
   */
  static TimeStamp now();

  //! Returns a time stamp which lies \a msec ms in the future.
  static TimeStamp futureMSec(uint64_t msec);

  /*! Returns a time stamp parsed from an ISO 8601 basic UTC timestamp
   *  (YYYYMMDDTHHMMSS,fffffffff).
   */
  static TimeStamp fromIso8601BasicUTC(const String& str);

  //! Set the timestamp to the current system time.
  TimeStamp& fromNow();

  /*! Return a formatted time string.
   *  \note While TimeStamp uses a 64-bit unsigned integer to store
   *        the seconds, the time formatting methods only support
   *        32-bit signed integers and will therefore render
   *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
   *        incorrectly.
   */
  void strfTime(char* dest, size_t max_len, const char *format) const;
  /*! Return a formatted time string converted to the local timezone.
   *  \note While TimeStamp uses a 64-bit unsigned integer to store
   *        the seconds, the time formatting methods only support
   *        32-bit signed integers and will therefore render
   *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
   *        incorrectly.
   */
  void strfLocaltime(char* dest, size_t max_len, const char *format) const;
  /*! Return the TimeStamp as a string in ISO 8601 format, in the
   *  local timezone.
   *  \note While TimeStamp uses a 64-bit unsigned integer to store
   *        the seconds, the time formatting methods only support
   *        32-bit signed integers and will therefore render
   *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
   *        incorrectly.
   */
  String formatIso8601() const;
  /*! Return the TimeStamp as a string in ISO 8601 format, in UTC.
   *  \note While TimeStamp uses a 64-bit unsigned integer to store
   *        the seconds, the time formatting methods only support
   *        32-bit signed integers and will therefore render
   *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
   *        incorrectly.
   */
  String formatIso8601UTC() const;
  /*! Return the TimeStamp as a string in the ISO 8601 basic format
   *  (YYYYMMDDTHHMMSS,fffffffff), in the local timezone.
   *  \note While TimeStamp uses a 64-bit unsigned integer to store
   *        the seconds, the time formatting methods only support
   *        32-bit signed integers and will therefore render
   *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
   *        incorrectly.
   */
  String formatIso8601Basic() const;
  /*! Return the TimeStamp as a string in the ISO 8601 basic format
   *  (YYYYMMDDTHHMMSS,fffffffff), in UTC.
   *  \note While TimeStamp uses a 64-bit unsigned integer to store
   *        the seconds, the time formatting methods only support
   *        32-bit signed integers and will therefore render
   *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
   *        incorrectly.
   */
  String formatIso8601BasicUTC() const;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! This static function returns a TimeStamp that contains the
   *  current System time.
   *  \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE TimeStamp Now() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns a time stamp which lies \a msec ms in the future.
   *  \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE TimeStamp FutureMSec(uint64_t msec) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Set the timestamp to the current system time.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE TimeStamp& FromNow() ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE void Strftime(char* dest, size_t max_len, const char *format) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE void StrfLocaltime(char* dest, size_t max_len, const char *format) const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE String FormatIso8601() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

  //! Adds a TimeSpan.
  TimeStamp& operator += (const TimeSpan& span)
  {
    secs+=span.secs;
    nsecs+=span.nsecs;
    normalizeTime();
    return *this;
  }

  //! Substracts a TimeSpan.
  TimeStamp& operator -= (const TimeSpan& span)
  {
    secs-=span.secs;
    nsecs-=span.nsecs;
    normalizeTime();
    return *this;
  }

  /*! Compares two variables of type TimeStamp.
   *  \returns \c true if they are not equal.
   */
  bool operator != (const TimeStamp& other) const
  {
    return TimeBase::operator != (other);
  }

  /*! Compares two variables of type TimeStamp.
   *  \returns \c true if they are equal.
   */
  bool operator == (const TimeStamp& other) const
  {
    return TimeBase::operator == (other);
  }

  /*! Compares two variables of type TimeStamp.
   *  \returns \c true if the first one is earlier than the second
   *           one.
   */
  bool operator < (const TimeStamp& other) const
  {
    return TimeBase::operator < (other);
  }

  /*! Compares two variables of type TimeStamp.
   *  \returns \c true if the first one is later than the second one.
   */
  bool operator > (const TimeStamp& other) const
  {
    return TimeBase::operator > (other);
  }

  /*! Compares two variables of type TimeStamp.
   *  \returns \c true if the first one is earlier than or equal to
   *           the second one.
   */
  bool operator <= (const TimeStamp& other) const
  {
    return TimeBase::operator <= (other);
  }

  /*! Compares two variables of type TimeStamp.
   *  \returns \c true if the first one is later than or equal to the
   *           second one.
   */
  bool operator >= (const TimeStamp& other) const
  {
    return TimeBase::operator >= (other);
  }

  uint64_t tsSec() const { return secs; }
  uint32_t tsUSec() const { return nsecs/1000; }
  uint32_t tsNSec() const { return nsecs; }

  static TimeStamp maxTime()
  {
    return TimeStamp(TimeBase::maxTime());
  }

  using TimeBase::days;
  using TimeBase::hours;
  using TimeBase::minutes;
  using TimeBase::seconds;
  using TimeBase::milliSeconds;
  using TimeBase::microSeconds;
  using TimeBase::nanoSeconds;

  using TimeBase::timespec;
  using TimeBase::systemTimespec;

#ifdef _SYSTEM_DARWIN_
  using TimeBase::machTimespec;
#endif

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  ICL_CORE_VC_DEPRECATE_STYLE uint64_t TsSec() const ICL_CORE_GCC_DEPRECATE_STYLE;
  ICL_CORE_VC_DEPRECATE_STYLE uint32_t TsUSec() const ICL_CORE_GCC_DEPRECATE_STYLE;
  ICL_CORE_VC_DEPRECATE_STYLE uint32_t TsNSec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  static ICL_CORE_VC_DEPRECATE_STYLE TimeStamp MaxTime() ICL_CORE_GCC_DEPRECATE_STYLE;

  using TimeBase::Days;
  using TimeBase::Hours;
  using TimeBase::Minutes;
  using TimeBase::Seconds;
  using TimeBase::MilliSeconds;
  using TimeBase::MicroSeconds;
  using TimeBase::NanoSeconds;

  using TimeBase::Timespec;
  using TimeBase::SystemTimespec;

#ifdef _SYSTEM_DARWIN_
  using TimeBase::MachTimespec;
#endif

#endif
  /////////////////////////////////////////////////

  static const TimeStamp cZERO;

private:
  TimeStamp(const TimeBase &base)
    : TimeBase(base)
  { }

  TimeStamp& operator += (const TimeStamp& other);
};


inline TimeStamp operator + (const TimeSpan& span, const TimeStamp& time)
{
  TimeStamp a(time);
  return a += span;
}

inline TimeStamp operator + (const TimeStamp& time, const TimeSpan& span)
{
  TimeStamp a(time);
  return a += span;
}

inline TimeStamp operator - (const TimeStamp& time, const TimeSpan& span)
{
  TimeStamp a(time);
  return a -= span;
}

inline TimeSpan operator - (const TimeStamp& time_1, const TimeStamp& time_2)
{
  TimeSpan a(time_1.tsSec()-time_2.tsSec(), time_1.tsNSec()-time_2.tsNSec());
  return a;
}

}
#endif
