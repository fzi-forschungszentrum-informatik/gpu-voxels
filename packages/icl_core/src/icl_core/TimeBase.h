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
 * \brief   Contains TimeBase
 *
 * \b tTime
 *
 * Contains the definitions of a generic interface  to the system time
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_TIME_BASE_H_INCLUDED
#define ICL_CORE_TIME_BASE_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/ImportExport.h"
#include "icl_core/os.h"

#ifdef _IC_BUILDER_HAS_TIME_H_
# include <time.h>
#endif
#ifdef _IC_BUILDER_HAS_SYS_TIME_H_
# include <sys/time.h>
#endif

#ifdef _SYSTEM_DARWIN_
# include <mach/clock_types.h>
#endif

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

//! Repesents time values.
/*!
 * You should not use this class directly. Use the subclasses
 * TimeStamp and TimeSpan instead.
 */
class ICL_CORE_IMPORT_EXPORT TimeBase
{
public:
  //! Adds a TimeSpan.
  TimeBase& operator += (const TimeBase& span);

  //! Substracts a TimeSpan.
  TimeBase& operator -= (const TimeBase& span);

  /*! Compares two variables of type TimeBase.
   *  \returns \c true if they are not equal.
   */
  bool operator != (const TimeBase& other) const;

  /*! Compares two variables of type TimeBase.
   *  \returns \c true if they are equal.
   */
  bool operator == (const TimeBase& other) const;

  /*! Compares two variables of type TimeBase.
   *  \returns \c true if the first one is earlier than the second
   *           one.
   */
  bool operator < (const TimeBase& other) const;

  /*! Compares two variables of type TimeBase.
   *  \returns \c true if the first one is later than the second one.
   */
  bool operator > (const TimeBase& other) const;

  /*! Compares two variables of type TimeBase.
   *  \returns \c true if the first one is earlier than or equal to
   *           the second one.
   */
  bool operator <= (const TimeBase& other) const;

  /*! Compares two variables of type TimeBase.
   *  \returns \c true if the first one is later than or equal to the
   *           second one.
   */
  bool operator >= (const TimeBase& other) const;

  /*! Normalizes this time so that the nanosecond part is between 0
   *  and sign(sec)999999999.
   */
  void normalizeTime();

  //! Use this function if you want to express the time in days.
  int64_t days() const
  {
    return secs / 86400;
  }

  /*! Use this function if you want to express the time in hours,
   *  minutes and seconds.
   */
  int64_t hours() const
  {
    return secs / 3600 % 24;
  }

  /*! Use this function if you want to express the time in hours,
   *  minutes and seconds.
   */
  int64_t minutes() const
  {
    return (secs % 3600) / 60;
  }

  /*! Use this function if you want to express the time in
   *  hours, minutes and seconds.
   */
  int64_t seconds() const
  {
    return (secs % 3600) % 60;
  }

  /*! Use this function if you want to express the time in hours,
   *  minutes, seconds and milliseconds.
   */
  int32_t milliSeconds() const
  {
    return nsecs / 1000000;
  }

  /*! Use this function if you want to express the time in hours,
   *  minutes, seconds, milliseconds and microseconds.
   */
  int32_t microSeconds() const
  {
    return nsecs / 1000;
  }

  /*! Use this function if you want to express the time in hours,
   *  minutes, seconds, milliseconds, microseconds and nanoseconds.
   */
  int32_t nanoSeconds() const
  {
    return nsecs % 1000;
  }

  //! Returns the second part of this time.
  int64_t tbSec() const { return secs; }
  //! Returns the nanosecond part of this time.
  int32_t tbNSec() const { return nsecs; }

  /*! Convert to <tt>struct timespec</tt>.  The base time for this
   *  conversion is the global clock (i.e. 0 = 1970-01-01 00:00:00).
   *
   *  \note If the native time_t is only 32 bits long then the
   *  returned timespec may not equal the original TimeBase.
   */
  struct timespec timespec() const;

  /*! Convert to <tt>struct timespec</tt>.  The base time for this
   *  conversion is the system clock, which may differ from the world
   *  clock on some systems.
   */
  struct timespec systemTimespec() const;

#ifdef _SYSTEM_DARWIN_
  //! Convert to \c mach_timespec_t.
  mach_timespec_t machTimespec() const;
#endif

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Normalizes this time so that the nanosecond part is between 0
   *  and sign(sec)999999999.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void NormalizeTime() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in days.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int64_t Days() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in hours,
   *  minutes and seconds.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int64_t Hours() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in hours,
   *  minutes and seconds.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int64_t Minutes() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in hours,
   *  minutes and seconds.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int64_t Seconds() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in hours,
   *  minutes, seconds and milliseconds.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int32_t MilliSeconds() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in hours,
   *  minutex, seconds, milliseconds and microseconds.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int32_t MicroSeconds() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Use this function if you want to express the time in hours,
   *  minutex, seconds, milliseconds, microseconds and nanoseconds.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int32_t NanoSeconds() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the second part of this time.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int64_t TbSec() const ICL_CORE_GCC_DEPRECATE_STYLE;
  /*! Returns the nanosecond part of this time.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE int32_t TbNSec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Convert to <tt>struct timespec</tt>.  The base time for this
   *  conversion is the global clock (i.e. 0 = 1970-01-01 00:00:00).
   *
   *  \note If the native time_t is only 32 bits long then the
   *  returned timespec may not equal the original TimeBase.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE struct timespec Timespec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Convert to <tt>struct timespec</tt>. The base time for this
   *  conversion is the system clock, which may differ from the world
   *  clock on some systems.  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE struct timespec SystemTimespec() const ICL_CORE_GCC_DEPRECATE_STYLE;

#ifdef _SYSTEM_DARWIN_
  /*! Convert to \c mach_timespec_t.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE mach_timespec_t MachTimespec() const ICL_CORE_GCC_DEPRECATE_STYLE;
#endif

#endif
  /////////////////////////////////////////////////

protected:

  static TimeBase maxTime();

  explicit TimeBase(int64_t secs=0, int32_t nsecs=0);

  TimeBase(const struct timespec& time);

  void fromTimespec(const struct timespec& time);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  static ICL_CORE_VC_DEPRECATE_STYLE TimeBase MaxTime() ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE void FromTimespec(const struct timespec& time) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

  int64_t secs;
  int32_t nsecs;
};

}

#endif
