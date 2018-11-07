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
 * \author  Florian Kuhnt <kuhnt@fzi.de>
 * \date    2014-05-07
 *
 * \brief   Contains ConfigPositionalParameter
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_POSITIONAL_PARAMETER_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_POSITIONAL_PARAMETER_H_INCLUDED

#include "icl_core/BaseTypes.h"
#include "icl_core/Vector.h"
#include "icl_core_config/ImportExport.h"
#include "icl_core_config/GetoptPositionalParameter.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace config {

/*! Contains information about how to handle a specific positional
 *  commandline parameter and how to map it into a configuration
 *  parameter.
 *
 *  The option value is stored in the specified config key.
 */
class ICL_CORE_CONFIG_IMPORT_EXPORT ConfigPositionalParameter : public GetoptPositionalParameter
{
public:
  /*! Create a new config postional parameter.
   *
   *  \param name The name of the parameter.
   *  \param config_key The configuration key in which the option
   *         value should be stored.
   *  \param help A help text that will be used in the generic help.
   *  \param is_optional Iff the parameter is an optional parameter.
   *  \param default_value The default value to be set, if it has
   *         neither been set in the config file and on the
   *         commandline.
   *
   *  \see GetoptPositionalParameter
   */
  ConfigPositionalParameter(const icl_core::String& name,
                            const icl_core::String& config_key,
                            const icl_core::String& help,
                            const bool is_optional=false,
                            const icl_core::String& default_value = "");

  /*! Get the configuration key in which the option should be stored.
   */
  icl_core::String configKey() const { return m_config_key; }

  /*! Check if a default value has been set.
   */
  bool hasDefaultValue() const { return !m_default_value.empty(); }

  /*! Get the default value of the configuration parameter.
   */
  icl_core::String defaultValue() const { return m_default_value; }

private:
  icl_core::String m_config_key;
  icl_core::String m_default_value;
};

typedef icl_core::Vector<ConfigPositionalParameter> ConfigPositionalParameterList;

}
}

#endif
