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
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2010-04-08
 *
 */
//----------------------------------------------------------------------
#include <icl_core/BaseTypes.h>
#include <icl_core/SchemeParser.h>

#include <boost/test/unit_test.hpp>
#include <list>

using icl_core::SchemeParser;

BOOST_AUTO_TEST_SUITE(ts_tParser)

BOOST_AUTO_TEST_CASE(TestStaticParser)
{
  BOOST_SPIRIT_NAMESPACE::parse_info<> info;
  icl_core::Scheme scheme;
  scheme.scheme_type = icl_core::OtherScheme;
  icl_core::String input = "file:///path/to/file/data.abc#anchor123?foo=bar&test=me";

  BOOST_CHECK_EQUAL(icl_core::SchemeParser::parseScheme(input, scheme, info), true);
  BOOST_CHECK_EQUAL(scheme.scheme_type, icl_core::FileScheme);
  BOOST_CHECK_EQUAL(scheme.specifier, "/path/to/file/data.abc");
  BOOST_CHECK_EQUAL(scheme.anchor, "anchor123");
  BOOST_CHECK_EQUAL(scheme.queries[0].name, "foo");
  BOOST_CHECK_EQUAL(scheme.queries[0].value, "bar");
  BOOST_CHECK_EQUAL(scheme.queries[1].name, "test");
  BOOST_CHECK_EQUAL(scheme.queries[1].value, "me");
}

BOOST_AUTO_TEST_CASE(TestParserClass)
{
  std::list<icl_core::String> input_list;

  input_list.push_back("file:///path/to/my_file/data.tof");
  input_list.push_back("file:///path/to/my file/my data.tof");
  input_list.push_back("file:///path/to/file/data.tof?offset=2000&skip_frames=2");

  // Test RawScheme
  input_list.push_back("raw:///path/to/file/data.tof");

  // Test OutputScheme
  input_list.push_back("output://video");

  // Test CameraScheme
  input_list.push_back("camera://dragonfly1?guid=0123456");
  input_list.push_back("camera://vision_a3?ip=192.168.95.123#&port=1919");

  // Test WithoutScheme
  input_list.push_back("/path/to/file/data.tof");

  // Test WithoutPath
  input_list.push_back("data.tof");

  std::list<icl_core::String>::iterator iter = input_list.begin();

  icl_core::SchemeParser parser;

  while (iter != input_list.end())
  {
    BOOST_CHECK_EQUAL(parser.parseScheme(*iter), true);
    ++iter;
  }
}


BOOST_AUTO_TEST_SUITE_END()
