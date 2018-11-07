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
#include "icl_core_config/ConfigParameter.h"

namespace icl_core {
namespace config {

ConfigParameter::ConfigParameter(const icl_core::String& option, const icl_core::String& short_option,
                                 const icl_core::String& config_key, const icl_core::String& help,
                                 const icl_core::String& default_value)
  : GetoptParameter(option, short_option,
                    default_value.empty() ? help : help + "\n(defaults to " + default_value + ")"),
    m_config_key(config_key),
    m_default_value(default_value)
{ }

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Get the configuration key in which the option should be stored.
 *  \deprecated Obsolete coding style.
 */
icl_core::String ConfigParameter::ConfigKey() const
{
  return configKey();
}

/*! Check if a default value has been set.
 *  \deprecated Obsolete coding style.
 */
bool ConfigParameter::HasDefaultValue() const
{
  return hasDefaultValue();
}

/*! Get the default value of the configuration parameter.
 *  \deprecated Obsolete coding style.
 */
icl_core::String ConfigParameter::DefaultValue() const
{
  return defaultValue();
}

#endif
/////////////////////////////////////////////////

}
}
