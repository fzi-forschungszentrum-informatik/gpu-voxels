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
 * \date    2010-04-20
 *
 * \brief   Utility functions for the configuration framework.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_UTIL_H_INCLUDED
#define ICL_CORE_CONFIG_UTIL_H_INCLUDED

#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <boost/algorithm/string.hpp>

namespace icl_core {
namespace config {
//! Namespace for internal implementation details.
namespace impl {

/*! Lexical cast which handles hexadecimal and octal numbers
 *  correctly.
 */
template <typename T, typename U>
T hexical_cast(U input)
{
  std::stringstream interpreter;
  interpreter.setf(std::ios::fmtflags(), std::ios::basefield);
  interpreter << input;
  T result;
  interpreter >> result;
  return result;
}

/*! Lexical cast to bool.  Handles hexadecimal and octal numbers
 *  correctly like other integer values.
 *  \li The values 1, "yes" and "true" (in any capitalization) are
 *      interpreted as true.
 *  \li Any other value is interpreted as false.
 */
template <typename U>
bool bool_cast(U input)
{
  std::stringstream interpreter;
  interpreter.setf(std::ios::fmtflags(), std::ios::basefield);
  interpreter << input;
  std::string result;
  interpreter >> result;
  boost::algorithm::to_lower(result);
  if (result == "1" || result == "yes" || result == "true")
  {
    return true;
  }
  else
  {
    return false;
  }
}

/*! Lexical cast to bool.  Handles hexadecimal and octal numbers
 *  correctly like other integer values.
 *  \li The values 1, "yes" and "true" (in any capitalization) are
 *      interpreted as true.
 *  \li The values 0, "no" and "false" (in any capitalization) are
 *      interpreted as false.
 *  \li For any other value, std::invalid_error is thrown.
 */
template <typename U>
bool strict_bool_cast(U input)
{
  std::stringstream interpreter;
  interpreter.setf(std::ios::fmtflags(), std::ios::basefield);
  interpreter << input;
  std::string result;
  interpreter >> result;
  boost::algorithm::to_lower(result);
  if (result == "1" || result == "yes" || result == "true")
  {
    return true;
  }
  else if (result == "0" || result == "no" || result == "false")
  {
    return false;
  }
  else
  {
    throw std::invalid_argument("Not a boolean value: " + result);
  }
}

}
}
}

#endif
