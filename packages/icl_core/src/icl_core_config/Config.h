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
 * \date    2008-11-01
 *
 * \brief   Base header file for the configuration framework.
 *
 * Contains convenience functions to access the ConfigManager singleton's functionality.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_H_INCLUDED

#include <boost/foreach.hpp>

#include <icl_core/BaseTypes.h>
#include <icl_core/EnumHelper.h>
#include <icl_core/StringHelper.h>
#include <icl_core/TemplateHelper.h>
#include <icl_core_logging/Logging.h>

#include "icl_core_config/ImportExport.h"
#include "icl_core_config/ConfigIterator.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigParameter.h"
#include "icl_core_config/ConfigValues.h"
#include "icl_core_config/GetoptParser.h"
#include "icl_core_config/GetoptParameter.h"
#include "icl_core_config/Util.h"

#ifdef _IC_BUILDER_OPENSPLICEDDS_
# include "dds_dcps.h"
#endif

#ifdef _IC_BUILDER_DEPRECATED_STYLE_

// -- START Deprecated compatibility headers --
#include "icl_core_config/tConfig.h"
#include "icl_core_config/tConfigIterator.h"
#include "icl_core_config/tConfigParameter.h"
#include "icl_core_config/tConfigValues.h"
#include "icl_core_config/tGetopt.h"
#include "icl_core_config/tGetoptParameter.h"
// -- END Deprecated compatibility headers --

# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
//! Framework for processing configuration files.
namespace config {

extern ICL_CORE_CONFIG_IMPORT_EXPORT const char * CONFIGFILE_CONFIG_KEY;

ICL_CORE_CONFIG_IMPORT_EXPORT void dump();

ICL_CORE_CONFIG_IMPORT_EXPORT void debugOutCommandLine(int argc, char *argv[]);

ICL_CORE_CONFIG_IMPORT_EXPORT ConfigIterator find(const icl_core::String& query);

//! Gets the value for the specified \a key from the configuration.
template <typename T>
bool get(const icl_core::String& key, typename icl_core::ConvertToRef<T>::ToRef value);

//! Gets the value for the specified \a key from the configuration.
template <typename T>
bool get(const char *key, typename icl_core::ConvertToRef<T>::ToRef value)
{
  return get<T>(icl_core::String(key), value);
}

//! Gets the value for the specified \a key from the configuration.
template <typename T>
bool get(const icl_core::String& key, typename icl_core::ConvertToRef<T>::ToRef value)
{
  icl_core::String str_value;
  if (ConfigManager::instance().get(key, str_value))
  {
    try
    {
      value = impl::hexical_cast<T>(str_value);
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

//! Template specialization for icl_core::String.
template <>
inline
bool get<icl_core::String>(const icl_core::String& key, icl_core::String& value)
{
  return ConfigManager::instance().get(key, value);
}

//! Template specialization for boolean values.
template <>
inline
bool get<bool>(const icl_core::String& key, bool& value)
{
  icl_core::String str_value;
  if (ConfigManager::instance().get(key, str_value))
  {
    str_value = toLower(str_value);
    if (str_value == "0" || str_value == "no" || str_value == "false")
    {
      value = false;
      return true;
    }
    else if (str_value == "1" || str_value == "yes" || str_value == "true")
    {
      value = true;
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

#ifdef _IC_BUILDER_OPENSPLICEDDS_
//! Template specialization for DDS_string values.
template<>
inline
bool get<DDS::String>(const icl_core::String& key, DDS::String& value)
{
  icl_core::String str_value;
  if (ConfigManager::instance().get(key, str_value))
  {
    value = DDS::string_dup(str_value.c_str());
    return true;
  }
  else
  {
    return false;
  }
}
#endif

template <typename T>
bool get(const icl_core::String& key, typename icl_core::ConvertToRef<T>::ToRef value,
         const char *descriptions[], const char *end_marker = NULL)
{
  icl_core::String str_value;
  if (ConfigManager::instance().get(key, str_value))
  {
    int32_t raw_value;
    if (icl_core::string2Enum(str_value, raw_value, descriptions, end_marker))
    {
      value = T(raw_value);
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

template <typename T>
bool get(const char *key, typename icl_core::ConvertToRef<T>::ToRef value,
         const char *descriptions[], const char *end_marker = NULL)
{
  return get<T>(icl_core::String(key), value, descriptions, end_marker);
}

template <typename T>
T getDefault(const icl_core::String& key, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{
  T value = default_value;
  get<T> (key, value);
  return value;
}

template <typename T>
T getDefault(const char *key, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{
  return getDefault<T> (icl_core::String(key), default_value);
}

template <> inline
icl_core::String getDefault<icl_core::String>(const icl_core::String& key,
                                              const icl_core::String& default_value)
{
  icl_core::String value = default_value;
  get<icl_core::String>(key, value);
  return value;
}

#ifdef _IC_BUILDER_OPENSPLICEDDS_
template<>
inline
DDS::String getDefault<DDS::String>(const icl_core::String& key,
                                    const DDS::String& default_value)
{
  DDS::String value = default_value;
  get<DDS::String>(key, value);
  return value;
}
#endif

/*! Get configuration parameters in batch mode. Returns \c true on
 *  success.  If \a report_error is \c true, writes an error message
 *  for each failed configuration parameter.  Returns \c false if any
 *  parameter failed.  Optionally deletes the contents of the \a
 *  config_values array.
 */
inline
bool get(const ConfigValues config_values, icl_core::logging::LogStream& log_stream,
         bool cleanup = true, bool report_error = true)
{
  // Read the configuration parameters.
  bool result = true;
  const impl::ConfigValueIface *const*config = config_values;
  while (*config != NULL)
  {
    if ((*config)->get())
    {
      SLOGGING_TRACE(log_stream, "Read configuration parameter \""
                     << (*config)->key() << "\" = \"" << (*config)->stringValue()
                     << "\"." << icl_core::logging::endl);
    }
    else
    {
      if (report_error)
      {
        SLOGGING_ERROR(log_stream, "Error reading configuration parameter \""
                       << (*config)->key() << "\"!" << icl_core::logging::endl);
      }
      else
      {
        SLOGGING_TRACE(log_stream, "Could not read configuration parameter \""
                       << (*config)->key() << "\"." << icl_core::logging::endl);
      }
      result = false;
    }
    ++config;
  }

  // Cleanup!
  if (cleanup)
  {
    config = config_values;
    while (*config != NULL)
    {
      delete *config;
      ++config;
    }
  }

  return result;
}

/*! Get configuration parameters in batch mode. Returns \c true on
 *  success.  If \a report_error is \c true, writes an error message
 *  for each failed configuration parameter.  Returns \c false if any
 *  parameter failed.  Optionally deletes the contents of the \a
 *  config_values array.
 */
inline
bool get(std::string config_prefix,
         ConfigValueList config_values, icl_core::logging::LogStream& log_stream,
         bool cleanup = true, bool report_error = true)
{
  /* Remark: config_values has to be passed by value, not by reference.
   *         Otherwise boost::assign::list_of() can not work correctly.
   */

  // Add a trailing slash, if necessary.
  if (config_prefix != "" && config_prefix[config_prefix.length() - 1] != '/')
  {
    config_prefix = config_prefix + "/";
  }

  // Read the configuration parameters.
  bool result = false;
  bool error = false;
  BOOST_FOREACH(impl::ConfigValueIface const * config, config_values)
  {
    if (config->get(config_prefix, log_stream))
    {
      SLOGGING_TRACE(log_stream, "Read configuration parameter \""
                     << config_prefix << config->key() << "\" = \"" << config->stringValue()
                     << "\"." << icl_core::logging::endl);
    }
    else
    {
      if (report_error)
      {
        SLOGGING_ERROR(log_stream, "Error reading configuration parameter \""
                       << config_prefix << config->key() << "\"!" << icl_core::logging::endl);
      }
      else
      {
        SLOGGING_TRACE(log_stream, "Could not read configuration parameter \""
                       << config_prefix << config->key() << "\"." << icl_core::logging::endl);
      }
      error = true;
    }
    result = true;
  }

  if (error)
  {
    result = false;
  }

  // Cleanup!
  if (cleanup)
  {
    BOOST_FOREACH(impl::ConfigValueIface * config, config_values)
    {
      delete config;
    }
  }

  return result;
}

inline
bool get(ConfigValueList config_values, icl_core::logging::LogStream& log_stream,
         bool cleanup = true, bool report_error = true)
{
  return get("", config_values, log_stream, cleanup, report_error);
}

template <typename T>
void setValue(const icl_core::String& key, typename icl_core::ConvertToRef<T>::ToConstRef value)
{
  ConfigManager::instance().setValue<T>(key, value);
}

template <typename T>
void setValue(const char *key, typename icl_core::ConvertToRef<T>::ToConstRef value)
{
  ConfigManager::instance().setValue<T>(icl_core::String(key), value);
}

inline
void setValue(const icl_core::String& key, const icl_core::String& value)
{
  setValue<icl_core::String>(key, value);
}

inline
bool paramOptPresent(const icl_core::String& name)
{
  return Getopt::instance().paramOptPresent(name);
}

template <typename T>
bool paramOpt(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToRef value)
{
  Getopt& getopt = Getopt::instance();
  if (getopt.paramOptPresent(name))
  {
    try
    {
      value = impl::hexical_cast<T>(getopt.paramOpt(name));
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
template <typename T>
bool paramOpt(const char *name, typename icl_core::ConvertToRef<T>::ToRef value)
{
  return paramOpt<T>(icl_core::String(name), value);
}

template <typename T>
bool paramOpt(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToRef value,
              const char *descriptions[], const char *end_marker = NULL)
{
  Getopt& getopt = Getopt::instance();
  if (getopt.paramOptPresent(name))
  {
    icl_core::String str_value = getopt.paramOpt(name);
    int32_t raw_value;
    if (icl_core::string2Enum(str_value, raw_value, descriptions, end_marker))
    {
      value = T(raw_value);
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

template <typename T>
bool paramOpt(const char *name, typename icl_core::ConvertToRef<T>::ToRef value,
              const char *descriptions[], const char *end_marker = NULL)
{
  return paramOpt<T>(icl_core::String(name), value, descriptions, end_marker);
}

template <typename T>
T paramOptDefault(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{
  Getopt& getopt = Getopt::instance();
  if (getopt.paramOptPresent(name))
  {
    try
    {
      return impl::hexical_cast<T>(getopt.paramOpt(name));
    }
    catch (...)
    {
      return default_value;
    }
  }
  else
  {
    return default_value;
  }
}

template <typename T>
T paramOptDefault(const char *name, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{
  return paramOptDefault<T>(icl_core::String(name), default_value);
}

template <typename T>
bool paramNonOpt(size_t index, typename icl_core::ConvertToRef<T>::ToRef value)
{
  Getopt& getopt = Getopt::instance();
  if (index < getopt.paramNonOptCount())
  {
    try
    {
      value = impl::hexical_cast<T>(getopt.paramNonOpt(index));
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

template <typename T>
bool paramNonOpt(size_t index, typename icl_core::ConvertToRef<T>::ToRef value,
                 const char *descriptions[], const char *end_marker = NULL)
{
  Getopt& getopt = Getopt::instance();
  if (index < getopt.paramNonOptCount())
  {
    icl_core::String str_value = getopt.paramNonOpt(index);
    int32_t raw_value;
    if (icl_core::string2Enum(str_value, raw_value, descriptions, end_marker))
    {
      value = T(raw_value);
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

inline icl_core::String paramNonOpt(size_t index)
{
  return Getopt::instance().paramNonOpt(index);
}

inline size_t extraCmdParamCount()
{
  return Getopt::instance().extraCmdParamCount();
}

inline icl_core::String extraCmdParam(size_t index)
{
  return Getopt::instance().extraCmdParam(index);
}

inline void activateExtraCmdParams(const icl_core::String& delimiter = "--")
{
  Getopt::instance().activateExtraCmdParams(delimiter);
}

inline size_t paramNonOptCount()
{
  return Getopt::instance().paramNonOptCount();
}

inline void addParameter(const ConfigParameter& parameter)
{
  ConfigManager::instance().addParameter(parameter);
}

inline void addParameter(const ConfigParameterList& parameters)
{
  ConfigManager::instance().addParameter(parameters);
}

inline void addParameter(const ConfigPositionalParameter& parameter)
{
  ConfigManager::instance().addParameter(parameter);
}

inline void addParameter(const ConfigPositionalParameterList& parameters)
{
  ConfigManager::instance().addParameter(parameters);
}

inline void addParameter(const GetoptParameter& parameter)
{
  Getopt::instance().addParameter(parameter);
}

inline void addParameter(const GetoptParameterList& parameters)
{
  Getopt::instance().addParameter(parameters);
}

inline void addParameter(const GetoptPositionalParameter& parameter)
{
  Getopt::instance().addParameter(parameter);
}

inline void addParameter(const GetoptPositionalParameterList& parameters)
{
  Getopt::instance().addParameter(parameters);
}

inline void setProgramVersion(icl_core::String const & version)
{
  Getopt::instance().setProgramVersion(version);
}

inline void setProgramDescription(icl_core::String const & description)
{
  Getopt::instance().setProgramDescription(description);
}

inline void printHelp()
{
  Getopt::instance().printHelp();
}

ICL_CORE_CONFIG_IMPORT_EXPORT bool initialize(int& argc, char *argv[], bool remove_read_arguments);

ICL_CORE_CONFIG_IMPORT_EXPORT
bool initialize(int& argc, char *argv[],
                Getopt::CommandLineCleaning cleanup = Getopt::eCLC_None,
                Getopt::ParameterRegistrationCheck registration_check = Getopt::ePRC_Strict);

////////////// DEPRECATED VERSIONS //////////////
//#ifdef _IC_BUILDER_DEPRECATED_STYLE_

ICL_CORE_VC_DEPRECATE_STYLE
void Dump()
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void Dump()
{ dump(); }

ICL_CORE_VC_DEPRECATE_STYLE
void DebugOutCommandLine(int argc, char *argv[])
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void DebugOutCommandLine(int argc, char *argv[])
{ debugOutCommandLine(argc, argv); }

ICL_CORE_VC_DEPRECATE_STYLE
ConfigIterator Find(const icl_core::String& query)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline ConfigIterator Find(const icl_core::String& query)
{ return find(query); }

/*! Gets the value for the specified \a key from the configuration.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
bool Get(const icl_core::String& key,
         typename icl_core::ConvertToRef<T>::ToRef value) ICL_CORE_GCC_DEPRECATE_STYLE;

/*! Gets the value for the specified \a key from the configuration.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
bool Get(const char *key, typename icl_core::ConvertToRef<T>::ToRef value) ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool Get(const char *key, typename icl_core::ConvertToRef<T>::ToRef value)
{ return get<T>(key, value); }

/*! Gets the value for the specified \a key from the configuration.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
bool Get(const icl_core::String& key,
         typename icl_core::ConvertToRef<T>::ToRef value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T> inline
ICL_CORE_VC_DEPRECATE_STYLE bool Get(const icl_core::String& key,
                                     typename icl_core::ConvertToRef<T>::ToRef value)
{ return get<T>(key, value); }

/*! Template specialization for icl_core::String.
 *  \deprecated Obsolete coding style.
 */
template <>
bool Get<icl_core::String>(const icl_core::String& key,
                           icl_core::String& value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <> inline
ICL_CORE_VC_DEPRECATE_STYLE bool Get<icl_core::String>(const icl_core::String& key,
                                                       icl_core::String& value)
{ return get<icl_core::String>(key, value); }

/*! Template specialization for boolean values.
 *  \deprecated Obsolete coding style.
 */
template <>
bool Get<bool>(const icl_core::String& key, bool& value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <> inline
ICL_CORE_VC_DEPRECATE_STYLE bool Get<bool>(const icl_core::String& key, bool& value)
{ return get<bool>(key, value); }

template <typename T>
bool Get(const icl_core::String& key,
         typename icl_core::ConvertToRef<T>::ToRef value,
         const char *descriptions[],
         const char *end_marker = NULL)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool Get(const icl_core::String& key,
                                     typename icl_core::ConvertToRef<T>::ToRef value,
                                     const char *descriptions[],
                                     const char *end_marker)
{ return get<T>(key, value, descriptions, end_marker); }

template <typename T>
bool Get(const char *key,
         typename icl_core::ConvertToRef<T>::ToRef value,
         const char *descriptions[],
         const char *end_marker = NULL)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool Get(const char *key,
                                     typename icl_core::ConvertToRef<T>::ToRef value,
                                     const char *descriptions[],
                                     const char *end_marker)
{ return get<T>(key, value, descriptions, end_marker); }

template <typename T>
T GetDefault(const icl_core::String& key,
             typename icl_core::ConvertToRef<T>::ToConstRef default_value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE T GetDefault(const icl_core::String& key,
                                         typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{ return getDefault<T>(key, default_value); }

template <typename T>
T GetDefault(const char *key,
             typename icl_core::ConvertToRef<T>::ToConstRef default_value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE T GetDefault(const char *key,
                                         typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{ return getDefault<T>(key, default_value); }

template <>
icl_core::String GetDefault<icl_core::String>(const icl_core::String& key,
                                              const icl_core::String& default_value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <> inline
ICL_CORE_VC_DEPRECATE_STYLE
icl_core::String GetDefault<icl_core::String>(const icl_core::String& key,
                                              const icl_core::String& default_value)
{ return getDefault<icl_core::String>(key, default_value); }

/*! Get configuration parameters in batch mode.
 *  \deprecated Obsolete coding style.
 */
ICL_CORE_VC_DEPRECATE_STYLE
bool Get(const ConfigValues config_values,
         icl_core::logging::LogStream& log_stream, bool cleanup = true)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline bool Get(const ConfigValues config_values,
                icl_core::logging::LogStream& log_stream, bool cleanup)
{ return get(config_values, log_stream, cleanup); }

template <typename T>
void SetValue(const icl_core::String& key,
              typename icl_core::ConvertToRef<T>::ToConstRef value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE void SetValue(const icl_core::String& key,
                                          typename icl_core::ConvertToRef<T>::ToConstRef value)
{ setValue<T>(key, value); }

template <typename T>
void SetValue(const char *key,
              typename icl_core::ConvertToRef<T>::ToConstRef value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE void SetValue(const char *key,
                                          typename icl_core::ConvertToRef<T>::ToConstRef value)
{ setValue<T>(key, value); }

ICL_CORE_VC_DEPRECATE_STYLE
void SetValue(const icl_core::String& key, const icl_core::String& value)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void SetValue(const icl_core::String& key, const icl_core::String& value)
{ setValue(key, value); }

ICL_CORE_VC_DEPRECATE_STYLE
bool ParamOptPresent(const icl_core::String& name)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline bool ParamOptPresent(const icl_core::String& name)
{ return paramOptPresent(name); }

template <typename T>
bool ParamOpt(const icl_core::String& name,
              typename icl_core::ConvertToRef<T>::ToRef value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool ParamOpt(const icl_core::String& name,
                                          typename icl_core::ConvertToRef<T>::ToRef value)
{ return paramOpt<T>(name, value); }

template <typename T>
bool ParamOpt(const char *name, typename icl_core::ConvertToRef<T>::ToRef value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool ParamOpt(const char *name, typename icl_core::ConvertToRef<T>::ToRef value)
{ return paramOpt<T>(name, value); }

template <typename T>
bool ParamOpt(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToRef value,
              const char *descriptions[], const char *end_marker = NULL)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
bool ParamOpt(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToRef value,
              const char *descriptions[], const char *end_marker)
{ return paramOpt<T>(name, value, descriptions, end_marker); }

template <typename T>
bool ParamOpt(const char *name, typename icl_core::ConvertToRef<T>::ToRef value,
              const char *descriptions[], const char *end_marker = NULL)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
bool ParamOpt(const char *name, typename icl_core::ConvertToRef<T>::ToRef value,
              const char *descriptions[], const char *end_marker)
{ return paramOpt<T>(name, value, descriptions, end_marker); }

template <typename T>
T ParamOptDefault(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
T ParamOptDefault(const icl_core::String& name, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{ return paramOptDefault<T>(name, default_value); }

template <typename T>
T ParamOptDefault(const char *name, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
T ParamOptDefault(const char *name, typename icl_core::ConvertToRef<T>::ToConstRef default_value)
{ return paramOptDefault<T>(name, default_value); }

template <typename T>
bool ParamNonOpt(size_t index, typename icl_core::ConvertToRef<T>::ToRef value)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
bool ParamNonOpt(size_t index, typename icl_core::ConvertToRef<T>::ToRef value)
{ return paramNonOpt<T>(index, value); }

template <typename T>
bool ParamNonOpt(size_t index, typename icl_core::ConvertToRef<T>::ToRef value,
                 const char *descriptions[], const char *end_marker = NULL)
ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE
bool ParamNonOpt(size_t index, typename icl_core::ConvertToRef<T>::ToRef value,
                 const char *descriptions[], const char *end_marker)
{ return paramNonOpt<T>(index, value, descriptions, end_marker); }

ICL_CORE_VC_DEPRECATE_STYLE
icl_core::String ParamNonOpt(size_t index)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline icl_core::String ParamNonOpt(size_t index)
{ return paramNonOpt(index); }

ICL_CORE_VC_DEPRECATE_STYLE
size_t ExtraCmdParamCount()
ICL_CORE_GCC_DEPRECATE_STYLE;
inline size_t ExtraCmdParamCount()
{ return extraCmdParamCount(); }

ICL_CORE_VC_DEPRECATE_STYLE
icl_core::String ExtraCmdParam(size_t index)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline icl_core::String ExtraCmdParam(size_t index)
{ return extraCmdParam(index); }

ICL_CORE_VC_DEPRECATE_STYLE
void ActivateExtraCmdParams(const icl_core::String& delimiter = "--")
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void ActivateExtraCmdParams(const icl_core::String& delimiter)
{ activateExtraCmdParams(delimiter); }

ICL_CORE_VC_DEPRECATE_STYLE
size_t ParamNonOptCount()
ICL_CORE_GCC_DEPRECATE_STYLE;
inline size_t ParamNonOptCount()
{ return paramNonOptCount(); }

ICL_CORE_VC_DEPRECATE_STYLE
void AddParameter(const ConfigParameter& parameter)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void AddParameter(const ConfigParameter& parameter)
{ addParameter(parameter); }

ICL_CORE_VC_DEPRECATE_STYLE
void AddParameter(const ConfigParameterList& parameters)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void AddParameter(const ConfigParameterList& parameters)
{ addParameter(parameters); }

ICL_CORE_VC_DEPRECATE_STYLE
void AddParameter(const GetoptParameter& parameter)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void AddParameter(const GetoptParameter& parameter)
{ addParameter(parameter); }

ICL_CORE_VC_DEPRECATE_STYLE
void AddParameter(const GetoptParameterList& parameters)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void AddParameter(const GetoptParameterList& parameters)
{ addParameter(parameters); }

ICL_CORE_VC_DEPRECATE_STYLE
void PrintHelp()
ICL_CORE_GCC_DEPRECATE_STYLE;
inline void PrintHelp()
{ printHelp(); }

ICL_CORE_VC_DEPRECATE_STYLE
bool Initialize(int& argc, char *argv[], bool remove_read_arguments)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline bool Initialize(int& argc, char *argv[], bool remove_read_arguments)
{ return initialize(argc, argv, remove_read_arguments); }

ICL_CORE_VC_DEPRECATE_STYLE
bool Initialize(int& argc, char *argv[],
                Getopt::CommandLineCleaning cleanup = Getopt::eCLC_None,
                Getopt::ParameterRegistrationCheck registration_check = Getopt::ePRC_Strict)
ICL_CORE_GCC_DEPRECATE_STYLE;
inline bool Initialize(int& argc, char *argv[],
                Getopt::CommandLineCleaning cleanup,
                Getopt::ParameterRegistrationCheck registration_check)
{ return initialize(argc, argv, cleanup, registration_check); }

//#endif
/////////////////////////////////////////////////

}
}

#endif
