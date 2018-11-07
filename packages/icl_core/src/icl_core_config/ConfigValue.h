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
#ifndef ICL_CORE_CONFIG_CONFIG_VALUE_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_VALUE_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/TemplateHelper.h>

#include "icl_core_config/ConfigHelper.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigValueIface.h"
#include "icl_core_config/Util.h"

#ifdef _IC_BUILDER_OPENSPLICEDDS_
# include "dds_dcps.h"
# include "mapping/String.h"
#endif

# define CONFIG_VALUE(key, value)                                       \
  (new icl_core::config::ConfigValue<ICL_CORE_CONFIG_TYPEOF(value)>(key, value))


namespace icl_core {
namespace config {

/*! Typed "container" class for batch reading of configuration
 *  parameters.
 */
template <typename T>
class ConfigValue : public impl::ConfigValueIface
{
public:
  /*! Create a placeholder for later batch reading of configuration
   *  parameters.
   */
  ConfigValue(const icl_core::String& key,
              typename icl_core::ConvertToRef<T>::ToRef value)
    : m_key(key),
      m_value(value)
  { }

  /*! We need a virtual destructor!
   */
  virtual ~ConfigValue() { }

  /*! Actually read the configuration parameter.
   */
  virtual bool get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
  {
    if (ConfigManager::instance().get(prefix + m_key, m_str_value))
    {
      try
      {
        m_value = impl::hexical_cast<T>(m_str_value);
        return true;
      }
      catch (...)
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
};

template<>
inline
bool ConfigValue<bool>::get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
{
  bool result = false;
  if (ConfigManager::instance().get(prefix + m_key, m_str_value))
  {
    try
    {
      m_value = impl::strict_bool_cast(m_str_value);
      result = true;
    }
    catch (...)
    {
      result = false;
    }
  }
  else
  {
    result = false;
  }
  return result;
}

#ifdef _IC_BUILDER_OPENSPLICEDDS_
template<>
inline
bool ConfigValue<DDS::String>::get(std::string const & prefix, icl_core::logging::LogStream& log_stream) const
{
  bool result = false;
  if (ConfigManager::instance().get(prefix + m_key, m_str_value))
  {
    m_value = DDS::string_dup(m_str_value.c_str());
    result = true;
  }
  else
  {
    result = false;
  }
  return result;
}
template<>
inline
bool ConfigValue<DDS::String_mgr>::get(std::string const & prefix,
                                       icl_core::logging::LogStream& log_stream) const
{
  bool result = false;
  if (ConfigManager::instance().get(prefix + m_key, m_str_value))
  {
    m_value = DDS::string_dup(m_str_value.c_str());
    result = true;
  }
  else
  {
    result = false;
  }
  return result;
}
#endif

}}

#endif
