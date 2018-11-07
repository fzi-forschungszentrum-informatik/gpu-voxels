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
 * \date    2009-03-12
 */
//----------------------------------------------------------------------
#include "icl_core_config/GetoptParameter.h"

namespace icl_core {
namespace config {

GetoptParameter::GetoptParameter(const icl_core::String& option, const icl_core::String& short_option,
                                 const icl_core::String& help, bool is_prefix)
  : m_short_option(short_option),
    m_help(help),
    m_is_prefix(is_prefix)
{
  if (!option.empty() && *option.rbegin() == ':')
  {
    m_option = option.substr(0, option.length() - 1);
    m_has_value = true;
  }
  else
  {
    m_option = option;
    m_has_value = false;
  }
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Get the long option name.
 *  \deprecated Obsolete coding style.
 */
icl_core::String GetoptParameter::Option() const
{
  return option();
}
/*! Get the short option name.
 *  \deprecated Obsolete coding style.
 */
icl_core::String GetoptParameter::ShortOption() const
{
  return shortOption();
}
/*! Check if the option also expects a value.
 *  \deprecated Obsolete coding style.
 */
bool GetoptParameter::HasValue() const
{
  return hasValue();
}
/*! Get the help text.
 *  \deprecated Obsolete coding style.
 */
icl_core::String GetoptParameter::Help() const
{
  return help();
}

/*! Check if this is a prefix option.
 *  \deprecated Obsolete coding style.
 */
bool GetoptParameter::IsPrefixOption() const
{
  return isPrefixOption();
}

#endif
/////////////////////////////////////////////////

}
}
