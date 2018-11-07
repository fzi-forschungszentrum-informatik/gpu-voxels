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
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_ENUM_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_ENUM_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/EnumHelper.h>
#include <icl_core/TemplateHelper.h>

#include "icl_core_config/ConfigHelper.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigValueIface.h"

#define CONFIG_ENUM(key, value, descriptions)                           \
  (new icl_core::config::ConfigEnum<ICL_CORE_CONFIG_TYPEOF(value)>(key, value, descriptions))

namespace icl_core {
namespace config {

/*! Typed "container" class for batch reading of configuration parameters.
 */
template <typename T>
class ConfigEnum : public impl::ConfigValueIface
{
public:
  /*! Create a placeholder for later batch reading of configuration
   *  parameters.
   */
  ConfigEnum(const icl_core::String& key,
             typename icl_core::ConvertToRef<T>::ToRef value,
             const char * const *descriptions,
             const char *end_marker = NULL)
    : m_key(key),
      m_value(value),
      m_descriptions(descriptions),
      m_end_marker(end_marker)
  { }

  /*! We need a virtual destructor!
   */
  virtual ~ConfigEnum() {}

  /*! Actually read the configuration parameter.
   */
  virtual bool get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
  {
    if (ConfigManager::instance().get(prefix + m_key, m_str_value))
    {
      int32_t raw_value;
      if (icl_core::string2Enum(m_str_value, raw_value, m_descriptions, m_end_marker))
      {
        m_value = T(raw_value);
        return true;
      }
      else
      {
        return false;
      }
    }
    else
    {
      return false;
    }
  }

  /*! Return the configuration key.
   */
  virtual icl_core::String key() const
  {
    return m_key;
  }

  /*! Return the value as string.
   */
  virtual icl_core::String stringValue() const
  {
    return m_str_value;
  }

protected:
  icl_core::String m_key;
  mutable icl_core::String m_str_value;
  typename icl_core::ConvertToRef<T>::ToRef m_value;
  const char * const *m_descriptions;
  const char *m_end_marker;
};

}}

#endif
