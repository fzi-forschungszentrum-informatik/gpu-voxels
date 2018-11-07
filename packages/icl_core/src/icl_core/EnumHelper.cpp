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
 * \date    2009-03-11
 */
//----------------------------------------------------------------------
#include <string.h>

#include "icl_core/EnumHelper.h"

namespace icl_core {

bool string2Enum(const String& str, int32_t& value,
                 const char * const *descriptions, const char *end_marker)
{
  bool result = false;

  for (int32_t index = 0;
       ((end_marker == NULL) && (descriptions[index] != NULL))
         || ((end_marker != NULL) && (::strcmp(descriptions[index], end_marker) != 0));
       ++index)
  {
    // Return success if a matching description has been found.
    if (::strcmp(str.c_str(), descriptions[index]) == 0)
    {
      value = index;
      result = true;
    }
  }

  return result;
}

namespace impl {
template<typename T>
bool string2Enum(const String& str, T& value,
                 const std::vector<std::string>& descriptions)
{
  bool result = false;

  for (T index = 0; index < T(descriptions.size()); ++index)
  {
    // Return success if a matching description has been found.
    if (str == descriptions[std::size_t(index)])
    {
      value = index;
      result = true;
    }
  }

  return result;
}
}

bool string2Enum(const String& str, int32_t& value,
                 const std::vector<std::string>& descriptions)
{
  return impl::string2Enum<int32_t>(str, value, descriptions);
}

bool string2Enum(const String& str, int64_t& value,
                 const std::vector<std::string>& descriptions)
{
  return impl::string2Enum<int64_t>(str, value, descriptions);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

bool String2Enum(const String& str, int32_t& value,
                 const char * const *descriptions, const char *end_marker)
{
  return string2Enum(str, value, descriptions, end_marker);
}

#endif
/////////////////////////////////////////////////

}
