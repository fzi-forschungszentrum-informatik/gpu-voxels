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
#ifndef ICL_CORE_CONFIG_CONFIG_ENUM_DEFAULT_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_ENUM_DEFAULT_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/TemplateHelper.h>

#include "icl_core_config/ConfigHelper.h"
#include "icl_core_config/ConfigEnum.h"
#include "icl_core_config/Util.h"

#define CONFIG_ENUM_DEFAULT(key, value, default_value, descriptions)                            \
  (new icl_core::config::ConfigEnumDefault<ICL_CORE_CONFIG_TYPEOF(value)>(key, value, default_value, descriptions))

namespace icl_core {
namespace config {

/*! Typed "container" class for batch reading of configuration
 *  parameters with a default value.
 */
template <typename T>
class ConfigEnumDefault : public ConfigEnum<T>
{
public:
  /*! Create a placeholder for later batch reading of configuration
   *  parameters.
   */
  ConfigEnumDefault(const icl_core::String& key,
                    typename icl_core::ConvertToRef<T>::ToRef value,
                    typename icl_core::ConvertToRef<T>::ToConstRef default_value,
                    const char * const * descriptions,
                    const char * end_marker = NULL)
    : ConfigEnum<T>(key, value, descriptions, end_marker),
      m_default_value(default_value)
  { }

  /*! We need a virtual destructor!
   */
  virtual ~ConfigEnumDefault() {}

  /*! Actually read the configuration parameter.
   */
  virtual bool get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
  {
    if (!ConfigEnum<T>::get(prefix, log_stream))
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
