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
#ifndef ICL_CORE_TIME_SPAN_H_INCLUDED
#define ICL_CORE_TIME_SPAN_H_INCLUDED

#include <iostream>

#include "icl_core/ImportExport.h"
#include "icl_core/TimeBase.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

#ifdef _IC_BUILDER_OPENSPLICEDDS_
struct iTimeSpan;
#endif

//! Repesents absolute times
/*! Use this class whenever you want to deal with times,
  as it provides a number of operators and functions.
*/
class ICL_CORE_IMPORT_EXPORT TimeSpan : protected TimeBase
{
  friend class TimeStamp;
public:

  /*! Constructs a time span from a second and nanosecond value.
   *  \param sec The second part of the newly constructed time span.
   *  \param nsec The nanosecond part of the time span.
   */
  explicit TimeSpan(int64_t sec=0, int32_t nsec=0);

  //! Constructs a time span from a struct timespec.
  TimeSpan(const struct timespec& time_span);

#ifdef _IC_BUILDER_OPENSPLICEDDS_
  //! Create a time span from its corresponding IDL datatype.
  TimeSpan(const iTimeSpan& time_span);
  //! Assign an IDL time span.
  TimeSpan& operator = (const iTimeSpan& time_span);
  //! Implicit conversion to an IDL time span.
  operator iTimeSpan ();
#endif

  //! Set the time span to \a sec seconds.
  TimeSpan& fromSec(int64_t sec);
  //! Set the time span to \a msec milliseconds.
  TimeSpan& fromMSec(int64_t msec);
  //! Set the time span to \a usec microseconds.
  TimeSpan& fromUSec(int64_t usec);

  //! Create a time span with \a sec seconds.
  static TimeSpan createFromSec(int64_t sec);
  //! Create a time span with \a msec milliseconds.
  static TimeSpan createFromMSec(int64_t msec);
  //! Create a time span with \a usec microseconds.
  static TimeSpan createFromUSec(int64_t usec);

  //! Adds a TimeSpan.
  TimeSpan& operator += (const TimeSpan& span);

  //! Substracts a TimeSpan.
  TimeSpan& operator -= (const TimeSpan& span);

  /*! Compares two variables of type TimeSpan.
   *  \returns \c true if they are not equal.
   */
  bool operator != (const TimeSpan& other) const;

  /*! Compares two variables of type TimeSpan.
   *  \returns \c true if they are equal.
   */
  bool operator == (const TimeSpan& other) const;

  /*! Compares two variables of type TimeSpan.
   *  \returns \c true if the first one is earlier than the second
   *           one.
   */
  bool operator < (const TimeSpan& other) const;

  /*! Compares two variables of type TimeSpan.
   *  \returns \c true if the first one is later than the second one.
   */
  bool operator > (const TimeSpan& other) const;

  /*! Compares two variables of type TimeSpan.
   *  \returns \c true if the first one is earlier than or equal to
   *           the second one.
   */
  bool operator <= (const TimeSpan& other) const;

  /*! Compares two variables of type TimeSpan.
   *  \returns \c true if the first one is later than or equal to
   *           the second one.
   */
  bool operator >= (const TimeSpan& other) const;

  int64_t tsSec() const;
  int32_t tsNSec() const;
  int32_t tsUSec() const;

  //! May result in an overflow if seconds are too large.
  int64_t toMSec() const;

  //! May result in an overflow if seconds are too large.
  int64_t toUSec() const;

  /*! Returns the timespan as nanoseconds. The conversion may result
   *  in an overflow if the seconds are too large.
   */
  int64_t toNSec() const;

  using TimeBase::days;
  using TimeBase::hours;
  using TimeBase::minutes;
  using TimeBase::seconds;
  using TimeBase::milliSeconds;
  using TimeBase::microSeconds;
  using TimeBase::nanoSeconds;

  using TimeBase::timespec;

  /* Remark: TimeBase has a systemTimespec() function, which is also
   * available in TimeStamp.  However, TimeSpan does not need this
   * function because it represents a relative time.  Therefore the
   * systemTimespec() function, which performs a transformation
   * between two "absolute zero points" (system and global time), does
   * not make any sense with the relative TimeSpan!
   */

#ifdef _SYSTEM_DARWIN_
  using TimeBase::machTimespec;
#endif

  static const TimeSpan cZERO;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  ICL_CORE_VC_DEPRECATE_STYLE TimeSpan& FromSec(int64_t sec) ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE TimeSpan& FromMSec(int64_t msec) ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE TimeSpan& FromUSec(int64_t usec) ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE int64_t TsSec() const ICL_CORE_GCC_DEPRECATE_STYLE;
  ICL_CORE_VC_DEPRECATE_STYLE int32_t TsNSec() const ICL_CORE_GCC_DEPRECATE_STYLE;
  ICL_CORE_VC_DEPRECATE_STYLE int32_t TsUSec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE int64_t ToMSec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE int64_t ToUSec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE int64_t ToNSec() const ICL_CORE_GCC_DEPRECATE_STYLE;

  using TimeBase::Days;
  using TimeBase::Hours;
  using TimeBase::Minutes;
  using TimeBase::Seconds;
  using TimeBase::MilliSeconds;
  using TimeBase::MicroSeconds;
  using TimeBase::NanoSeconds;

  using TimeBase::Timespec;

#ifdef _SYSTEM_DARWIN_
  using TimeBase::MachTimespec;
#endif

#endif
  /////////////////////////////////////////////////
};

inline TimeSpan operator + (const TimeSpan& left, const TimeSpan& right)
{
  TimeSpan a(left.tsSec() + right.tsSec(), left.tsNSec() + right.tsNSec());
  return a;
}

inline TimeSpan operator - (const TimeSpan& left, const TimeSpan& right)
{
  TimeSpan a(left.tsSec() - right.tsSec(), left.tsNSec() - right.tsNSec());
  return a;
}

inline TimeSpan operator / (const TimeSpan& span, double divisor)
{
  TimeSpan a(int64_t(span.tsSec() / divisor), int32_t(span.tsNSec() / divisor));
  return a;
}

inline TimeSpan operator * (const TimeSpan& span, double divisor)
{
  TimeSpan a(int64_t(span.tsSec() * divisor), int32_t(span.tsNSec() * divisor));
  return a;
}

inline TimeSpan abs(const TimeSpan& span)
{
  if (span < TimeSpan::cZERO)
  {
    return TimeSpan::cZERO - span;
  }
  else
  {
    return span;
  }
}

//! Write a time span to an STL output stream.
ICL_CORE_IMPORT_EXPORT std::ostream& operator << (std::ostream& stream, const TimeSpan& time_span);

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

inline TimeSpan Abs(const TimeSpan& span) ICL_CORE_GCC_DEPRECATE_STYLE;
ICL_CORE_VC_DEPRECATE_STYLE inline TimeSpan Abs(const TimeSpan& span)
{
  return abs(span);
}

#endif
/////////////////////////////////////////////////

}

#endif
