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
 * \date    2010-04-28
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_VALUE_DEFAULT_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_VALUE_DEFAULT_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/TemplateHelper.h>

#include "icl_core_config/ConfigHelper.h"
#include "icl_core_config/ConfigValue.h"
#include "icl_core_config/Util.h"

#define CONFIG_VALUE_DEFAULT(key, value, default_value)                 \
  (new icl_core::config::ConfigValueDefault<ICL_CORE_CONFIG_TYPEOF(value)>(key, value, default_value))

namespace icl_core {
namespace config {

/*! Typed "container" class for batch reading of configuration
 *  parameters with a default value.
 */
template <typename T>
class ConfigValueDefault : public ConfigValue<T>
{
public:
  /*! Create a placeholder for later batch reading of configuration
   *  parameters.
   */
  ConfigValueDefault(const icl_core::String& key,
                     typename icl_core::ConvertToRef<T>::ToRef value,
                     typename icl_core::ConvertToRef<T>::ToConstRef default_value)
    : ConfigValue<T>(key, value),
      m_default_value(default_value)
  { }

  /*! We need a virtual destructor!
   */
  virtual ~ConfigValueDefault() {}

  /*! Actually read the configuration parameter.
   */
  virtual bool get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
  {
    if (!ConfigValue<T>::get(prefix, log_stream))
    {
      this->m_value = m_default_value;
      this->m_str_value = impl::hexical_cast<icl_core::String>(this->m_value);
    }
    return true;
  }

private:
  typename icl_core::ConvertToRef<T>::ToConstRef m_default_value;
};

}}

#endif
