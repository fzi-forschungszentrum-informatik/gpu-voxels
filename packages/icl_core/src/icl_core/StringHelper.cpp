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
 * \date    2011-10-13
 *
 */
//----------------------------------------------------------------------
#include "icl_core/StringHelper.h"

namespace icl_core {

std::vector<String> split(const String& str, const String& delimiter)
{
  String s = str;
  String::size_type pos = 0;
  std::vector<String> substrings;
  if (s.empty())
  {
    substrings.push_back("");
    return substrings;
  }
  while ((pos = s.find(delimiter)) != String::npos)
  {
    substrings.push_back(s.substr(0, pos));
    s.erase(0, pos+delimiter.size());
  }
  if (!s.empty())
  {
    substrings.push_back(s);
  }
  return substrings;
}

String join(const std::vector<String>& substrings, const String& delimiter)
{
  String result;
  for (std::vector<String>::const_iterator it = substrings.begin(); it != substrings.end(); ++it)
  {
    if (it != substrings.begin())
    {
      result += delimiter;
    }
    result += *it;
  }
  return result;
}

String trim(String const & str)
{
  std::string result = "";

  std::string::size_type length = str.length();

  std::string::size_type trim_front = 0;
  while ((trim_front < length) && isspace(static_cast<unsigned char>(str[trim_front])))
  {
    ++trim_front;
  }

  std::string::size_type trim_end = length - 1;
  while ((trim_end > trim_front) && isspace(static_cast<unsigned char>(str[trim_end])))
  {
    --trim_end;
  }

  if (trim_front == length)
  {
    result = "";
  }
  else
  {
    result = str.substr(trim_front, trim_end - trim_front + 1);
  }

  return result;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

String Trim(String const & str)
{
  return trim(str);
}

#endif
/////////////////////////////////////////////////

}
