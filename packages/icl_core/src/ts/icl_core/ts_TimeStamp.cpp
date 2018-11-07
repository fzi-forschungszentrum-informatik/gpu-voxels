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
 * \author  Jan Oberlaender <oberlaender@fzi.de>
 * \date    2013-07-09
 *
 */
//----------------------------------------------------------------------
#include <icl_core/TimeStamp.h>

using icl_core::TimeStamp;

namespace icl_core {

std::ostream& operator << (std::ostream& os, const TimeStamp& t)
{
  return os << t.formatIso8601BasicUTC();
}

}

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ts_TimeStamp)

BOOST_AUTO_TEST_CASE(CheckISO8601RoundTrip)
{
  {
    TimeStamp t1(0, 0);
    TimeStamp t2 = TimeStamp::fromIso8601BasicUTC(t1.formatIso8601BasicUTC());
    BOOST_CHECK_EQUAL(t1, t2);
  }
  {
    TimeStamp t1(1000000000, 123456789);
    TimeStamp t2 = TimeStamp::fromIso8601BasicUTC(t1.formatIso8601BasicUTC());
    BOOST_CHECK_EQUAL(t1, t2);
  }
  {
    TimeStamp t1(2000000000, 12345);
    TimeStamp t2 = TimeStamp::fromIso8601BasicUTC(t1.formatIso8601BasicUTC());
    BOOST_CHECK_EQUAL(t1, t2);
  }
  {
    TimeStamp t1(2147483647, 999999999);
    TimeStamp t2 = TimeStamp::fromIso8601BasicUTC(t1.formatIso8601BasicUTC());
    BOOST_CHECK_EQUAL(t1, t2);
  }
  // Test some random values.
  for (uint64_t secs = 1; secs < 2000000000; secs += 72836471)
  {
    {
      TimeStamp t1(secs, 0);
      TimeStamp t2 = TimeStamp::fromIso8601BasicUTC(t1.formatIso8601BasicUTC());
      BOOST_CHECK_EQUAL(t1, t2);
    }
    {
      TimeStamp t1(secs, 999999999);
      TimeStamp t2 = TimeStamp::fromIso8601BasicUTC(t1.formatIso8601BasicUTC());
      BOOST_CHECK_EQUAL(t1, t2);
    }
  }
}

BOOST_AUTO_TEST_CASE(CheckISO8601UTCConversion)
{
  {
    std::string s = "19700101T000000,000000000";
    TimeStamp t1 = TimeStamp::fromIso8601BasicUTC(s);
    TimeStamp t2(0, 0);
    BOOST_CHECK_EQUAL(t1, t2);
  }
  {
    std::string s = "20040916T235959,25";
    TimeStamp t1 = TimeStamp::fromIso8601BasicUTC(s);
    TimeStamp t2(1095379199, 250000000);
    BOOST_CHECK_EQUAL(t1, t2);
  }
  {
    std::string s = "20130709T125530,123456789";
    TimeStamp t1 = TimeStamp::fromIso8601BasicUTC(s);
    TimeStamp t2(1373374530, 123456789);
    BOOST_CHECK_EQUAL(t1, t2);
  }
  {
    // 32-bit signed timestamp maximum.
    std::string s = "20380119T031407,999999999";
    TimeStamp t1 = TimeStamp::fromIso8601BasicUTC(s);
    TimeStamp t2(2147483647, 999999999);
    BOOST_CHECK_EQUAL(t1, t2);
  }
}

BOOST_AUTO_TEST_SUITE_END()
