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
 */
//----------------------------------------------------------------------
#include "SchemeParser.h"

#include <boost/foreach.hpp>

namespace icl_core
{

std::ostream &operator<<(std::ostream &stream, const Scheme &scheme)
{
  stream << scheme.scheme_name << scheme.specifier;

  bool first = true;
  BOOST_FOREACH(Query query, scheme.queries)
  {
    if ( first )
    {
      stream << "?";
      first = false;
    }
    else
    {
      stream << "&";
    }
    stream << query.name << "=" << query.value;
  }

  if (scheme.anchor.size() > 0)
  {
    stream << "#" << scheme.anchor;
  }

  return stream;
}


void SchemeFunction::operator () (char const* str, char const* end) const
{
  std::string name(str, end);
  for (size_t i = 0; i < name.size(); ++i)
  {
    name[i] = tolower(name[i]);
  }
  if (name == "file://")
  {
    m_scheme_handler->scheme_type = FileScheme;
  }
  else if (name == "http://")
  {
    m_scheme_handler->scheme_type = HttpScheme;
  }
  else if (name == "camera://")
  {
    m_scheme_handler->scheme_type = CameraScheme;
  }
  else if (name == "gps://")
  {
      m_scheme_handler->scheme_type = GpsScheme;
  }
  else
  {
    m_scheme_handler->scheme_type = OtherScheme;
  }
  m_scheme_handler->scheme_name = name;
}

void SpecifierFunction::operator () (char const* str, char const* end) const
{
  std::string name(str, end);
  m_scheme_handler->specifier = name;
}

void AnchorFunction::operator () (char const* str, char const* end) const
{
  std::string name(str, end);
  m_scheme_handler->anchor = name;
}

void QueryKeyFunction::operator () (char const* str, char const* end) const
{
  std::string name(str, end);
  Query query;
  query.name = name;
  m_queries->push_back(query);
}

void QueryValueFunction::operator () (char const* str, char const* end) const
{
  std::string value(str, end);
  if (m_queries->empty())
  {
    Query query;
    query.name = "";
    m_queries->push_back(query);
  }
  QueryList::reverse_iterator rit = m_queries->rbegin();
  assert(rit != m_queries->rend()); // Just to please Klocwork.
  rit->value = value;
}

SchemeParser::SchemeParser()
{
  m_scheme.scheme_type = icl_core::OtherScheme;
}

SchemeParser::~SchemeParser()
{
}

bool SchemeParser::parseScheme(const String &str)
{
  return this->parseScheme(str, m_scheme, m_info);
}

const BOOST_SPIRIT_NAMESPACE::parse_info<>& SchemeParser::getParseInfo() const
{
  return m_info;
}

const icl_core::Scheme& SchemeParser::getSchemeResult() const
{
  return m_scheme;
}

bool SchemeParser::parseScheme(const String &str, Scheme &scheme_handler,
                               BOOST_SPIRIT_NAMESPACE::parse_info<> &info)
{
  using BOOST_SPIRIT_NAMESPACE::rule;
  using BOOST_SPIRIT_NAMESPACE::alnum_p;
  using BOOST_SPIRIT_NAMESPACE::ch_p;
  using BOOST_SPIRIT_NAMESPACE::space_p;
  using BOOST_SPIRIT_NAMESPACE::str_p;
  using BOOST_SPIRIT_NAMESPACE::alpha_p;
  using BOOST_SPIRIT_NAMESPACE::anychar_p;

  SchemeFunction addScheme;
  addScheme.m_scheme_handler = &scheme_handler;

  SpecifierFunction addSpecifier;
  addSpecifier.m_scheme_handler = &scheme_handler;

  AnchorFunction addAnchor;
  addAnchor.m_scheme_handler = &scheme_handler;

  QueryKeyFunction addName;
  addName.m_queries = &scheme_handler.queries;

  QueryValueFunction addValue;
  addValue.m_queries = &scheme_handler.queries;

  // extended word rule, alphanumeric charactes, separated by _, -, ., or whitespace
  rule<> extword_p = +alnum_p >> *((ch_p('_') | ch_p('-') | ch_p('.') | space_p) >> +alnum_p);
  rule<> anchor_word = +alnum_p >> !(ch_p('-') >> +alnum_p);

  // special scheme characters
  rule<> scheme_ch = str_p("://");
  rule<> anchor_ch = ch_p('#');
  rule<> querystart_ch = ch_p('?');
  rule<> querydelim_ch = ch_p('&');

  // scheme, path and file rules
  rule<> scheme_p = +alpha_p >> *((ch_p('+')) >> +alpha_p) >> scheme_ch;                 // file+something://
  rule<> specifier_p = +(anychar_p - querystart_ch - anchor_ch);  /* almost everything between xyz://
                                                                   * and ?query=value or #anchor      */
  rule<> anchor_p = anchor_ch >> anchor_word[addAnchor];

  // query rules
  rule<> query_key = +alnum_p >> *(alnum_p | (ch_p('_') >> alnum_p));
  rule<> query_value = +(anychar_p - (querystart_ch | querydelim_ch | space_p));
  rule<> query_pair = query_key[addName] >> ch_p('=') >> query_value[addValue];
  rule<> query_p = querystart_ch >> query_pair >> *(querydelim_ch >> query_pair);

  // scheme rule
  rule<> scheme_rule = !scheme_p[addScheme] >> specifier_p[addSpecifier] >> !anchor_p >> !query_p;

  scheme_handler.queries.clear();
  info = parse(str.c_str(), scheme_rule);

  return info.full;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  bool SchemeParser::ParseScheme(const String &str)
  {
    return parseScheme(str);
  }

  const BOOST_SPIRIT_NAMESPACE::parse_info<> &SchemeParser::GetParseInfo() const
  {
    return getParseInfo();
  }

  const icl_core::Scheme &SchemeParser::GetSchemeResult() const
  {
    return getSchemeResult();
  }

  bool SchemeParser::ParseScheme(const String &str, Scheme &scheme_handler, BOOST_SPIRIT_NAMESPACE::parse_info<> &info)
  {
    return parseScheme(str, scheme_handler, info);
  }

#endif

}
