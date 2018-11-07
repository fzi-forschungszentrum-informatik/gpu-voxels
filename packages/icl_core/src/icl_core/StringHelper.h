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
 * \date    2009-07-07
 *
 * \brief   Contains helper functions for dealing with String.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_STRING_HELPER_H_INCLUDED
#define ICL_CORE_STRING_HELPER_H_INCLUDED

#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>
#include <ctype.h>
#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/ImportExport.h"
#include "icl_core/TemplateHelper.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

// all int types
template <typename T> String stringify(typename ConvertToRef<T>::ToConstRef x)
{
  std::ostringstream out;
  out << x;
  return out.str();
}

// bool
template <> inline String stringify<bool>(const bool& x)
{
  std::ostringstream out;
  out << std::boolalpha << x;
  return out.str();
}

template <> inline String stringify<double>(const double& x)
{
  const int sigdigits = std::numeric_limits<double>::digits10;
  std::ostringstream out;
  out << std::setprecision(sigdigits) << x;
  return out.str();
}

template <> inline String stringify<float>(const float& x)
{
  const int sigdigits = std::numeric_limits<float>::digits10;
  std::ostringstream out;
  out << std::setprecision(sigdigits) << x;
  return out.str();
}

template <> inline String stringify<long double>(const long double& x)
{
  const int sigdigits = std::numeric_limits<long double>::digits10;
  std::ostringstream out;
  out << std::setprecision(sigdigits) << x;
  return out.str();
}

inline String toLower(icl_core::String str)
{
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  return str;
}

inline String toUpper(icl_core::String str)
{
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
  return str;
}


template <typename T>
String padLeft(typename ConvertToRef<T>::ToConstRef x, size_t width, char pad_chr = ' ')
{
  return padLeft<String>(stringify<T>(x), width, pad_chr);
}

template <>
inline String padLeft<String>(const String& str, size_t width, char pad_chr)
{
  size_t str_len = str.length();
  if (str_len < width)
  {
    return String(width - str_len, pad_chr) + str;
  }
  else
  {
    return str;
  }
}

template <typename T>
String padRight(typename ConvertToRef<T>::ToConstRef x, size_t width, char pad_chr = ' ')
{
  return padRight<String>(stringify<T>(x), width, pad_chr);
}

template <>
inline String padRight<String>(const String& str, size_t width, char pad_chr)
{
  size_t str_len = str.length();
  if (str_len < width)
  {
    return str + String(width - str_len, pad_chr);
  }
  else
  {
    return str;
  }
}

/*! Simple string splitting method.  Given a string \a str, which
 *  contains substrings separated by \a delimiter, returns a vector of
 *  the substrings.  E.g., split("foo;bar;baz") returns a vector
 *  containing the elements "foo", "bar" and "baz".  If \a str is
 *  empty, returns a vector containing one element, the empty string.
 *  \param str The string to split.
 *  \param delimiter The delimiter string.
 */
ICL_CORE_IMPORT_EXPORT std::vector<String> split(const String& str, const String& delimiter);

/*! The reverse operation of split(const String&, const String&).
 *  Joins all the \a substrings, separated by \a delimiter.
 *  \param substrings The substrings to join.
 *  \param delimiter The delimiter string.
 */
ICL_CORE_IMPORT_EXPORT String join(const std::vector<String>& substrings, const String& delimiter);

ICL_CORE_IMPORT_EXPORT String trim(String const & str);

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

// all int types
template <typename T> String Stringify(typename ConvertToRef<T>::ToConstRef x) ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T> String Stringify(typename ConvertToRef<T>::ToConstRef x)
{
  return stringify<T>(x);
}

// bool
template <> inline String Stringify<bool>(const bool& x) ICL_CORE_GCC_DEPRECATE_STYLE;
template <> ICL_CORE_VC_DEPRECATE_STYLE inline String Stringify<bool>(const bool& x)
{
  return stringify<bool>(x);
}

template <> inline String Stringify<double>(const double& x) ICL_CORE_GCC_DEPRECATE_STYLE;
template <> ICL_CORE_VC_DEPRECATE_STYLE inline String Stringify<double>(const double& x)
{
  return stringify<double>(x);
}

template <> inline String Stringify<float>(const float& x) ICL_CORE_GCC_DEPRECATE_STYLE;
template <> ICL_CORE_VC_DEPRECATE_STYLE inline String Stringify<float>(const float& x)
{
  return stringify<float>(x);
}

template <> inline String Stringify<long double>(const long double& x) ICL_CORE_GCC_DEPRECATE_STYLE;
template <> ICL_CORE_VC_DEPRECATE_STYLE inline String Stringify<long double>(const long double& x)
{
  return stringify<long double>(x);
}

inline String Tolower(icl_core::String str) ICL_CORE_GCC_DEPRECATE_STYLE;
inline ICL_CORE_VC_DEPRECATE_STYLE String Tolower(icl_core::String str)
{
  return toLower(str);
}

inline String Toupper(icl_core::String str) ICL_CORE_GCC_DEPRECATE_STYLE;
inline ICL_CORE_VC_DEPRECATE_STYLE String Toupper(icl_core::String str)
{
  return toUpper(str);
}


template <typename T>
String PadLeft(typename ConvertToRef<T>::ToConstRef x, size_t width, char pad_chr = ' ')
  ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
String PadLeft(typename ConvertToRef<T>::ToConstRef x, size_t width, char pad_chr)
{
  return padLeft<T>(x, width, pad_chr);
}

template <>
inline String PadLeft<String>(const String& str, size_t width, char pad_chr)
  ICL_CORE_GCC_DEPRECATE_STYLE;
template <>
inline ICL_CORE_VC_DEPRECATE_STYLE
String PadLeft<String>(const String& str, size_t width, char pad_chr)
{
  return padLeft<String>(str, width, pad_chr);
}

template <typename T>
String PadRight(typename ConvertToRef<T>::ToConstRef x, size_t width, char pad_chr = ' ')
  ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
String PadRight(typename ConvertToRef<T>::ToConstRef x, size_t width, char pad_chr)
{
  return padRight<T>(x, width, pad_chr);
}

template <>
inline String PadRight<String>(const String& str, size_t width, char pad_chr)
  ICL_CORE_GCC_DEPRECATE_STYLE;
template <>
inline ICL_CORE_VC_DEPRECATE_STYLE
String PadRight<String>(const String& str, size_t width, char pad_chr)
{
  return padRight<String>(str, width, pad_chr);
}

ICL_CORE_VC_DEPRECATE_STYLE String Trim(String const & str) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

}

#endif
