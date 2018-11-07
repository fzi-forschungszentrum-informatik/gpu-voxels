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
 * \date    2007-12-04
 *
 * \brief   Contains ConfigManager.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_MANAGER_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_MANAGER_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/List.h>
#include <icl_core/Map.h>
#include <icl_core/TemplateHelper.h>
#include "icl_core_config/ConfigIterator.h"
#include "icl_core_config/ConfigParameter.h"
#include "icl_core_config/ConfigPositionalParameter.h"
#include "icl_core_config/ImportExport.h"
#include "icl_core_config/AttributeTree.h"

#include <boost/lexical_cast.hpp>

#include <cassert>

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

class TiXmlNode;

namespace icl_core {
namespace config {

class AttributeTree;
class ConfigObserver;

//! Class for handling configuration files.
/*!
 * ConfigManager is implemented as a singleton so that it
 * can be used from anywhere without the need to pass
 * a config object around.
 *
 * Before the configuration class can be used it has
 * to be initialized through a call to Initialize().
 * It will parse the command line and look for a
 * "-c [filename]" or "--configfile=[filename]" option.
 * ConfigManager will try to read the specified file and extract
 * all configuration attributes from it.
 *
 * Configuration files are XML files. The names of the XML
 * tags are used as the names of the configuration attributes
 * while the content text of the XML tags are used as
 * the attributes' values. Leading and trailing whitespace
 * is automatically removed from the values. Remark: The name
 * of the root element in the configuration file can be
 * chosen arbitrarily. It is not or interpreted from ConfigManager
 * in any way.
 *
 * Configuration attributes are retrieved using an XPath like
 * syntax. Hierarchical attribute names are separated by "/".
 */
class ICL_CORE_CONFIG_IMPORT_EXPORT ConfigManager: public icl_core::KeyValueDirectory<icl_core::String>
{
public:
  /*!
   * Get the singleton ConfigManager instance.
   */
  static ConfigManager& instance();

  /*!
   * Adds a commandline parameter.
   */
  void addParameter(const ConfigParameter& parameter);
  /*!
   * Adds a list of commandline parameters.
   */
  void addParameter(const ConfigParameterList& parameters);

  /*!
   * Adds a positional commandline parameter.
   */
  void addParameter(const ConfigPositionalParameter& parameter);
  /*!
   * Adds a list of positional commandline parameters.
   */
  void addParameter(const ConfigPositionalParameterList& parameters);

  /*!
   * Initializes ConfigManager. Reads the configuration file if
   * --configfile or -c has been specified on the commandline.
   * If no configuration file has been specified, the initialization
   * is treated as successful!
   *
   * \returns \a true if the initialization was successful, \a false
   *          otherwise. If the initialization fails, an error message
   *          will be printed to stderr.
   */
  bool initialize();

  /*!
   * Check if the configuration framework has already been initialized.
   */
  bool isInitialized() const
  {
    return m_initialized;
  }

  /*!
   * Dumps all configuration keys and the corresponding values
   * to stdout.
   */
  void dump() const;

  //! Add a key/value pair or change a value. In contrast to Insert, this method notifies observers
  template <class T>
  bool setValue(const icl_core::String &key, typename icl_core::ConvertToRef<T>::ToConstRef value)
  {
    icl_core::String string_value = boost::lexical_cast<icl_core::String>(value);

    if (key == "/configfile")
    {
      load(string_value);
    }

    bool result = insert(key, string_value);
    notify(key);
    return result;
  }

  /**! Register an observer which gets notified of changed key/value pairs
   *   @param observer The observer to add to the list of registered observers
   *   @param key The key to be notified of, or an empty string for all changes
   */
  void registerObserver(ConfigObserver *observer, const String &key = "");

  /*! Unregister an observer so it does not get notified of changes anymore.
   *  Normally you shouldn't need to call this as the destructor of config
   *  observers automatically calls it
   */
  void unregisterObserver(ConfigObserver *observer);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Get the singleton ConfigManager instance.
   *  \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE ConfigManager& Instance() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Adds a commandline parameter.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddParameter(const ConfigParameter& parameter) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Adds a list of commandline parameters.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddParameter(const ConfigParameterList& parameters) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Initializes ConfigManager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Initialize() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check if the configuration framework has already been
   *  initialized.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsInitialized() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Dumps all configuration keys and the corresponding values to
   *  stdout.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Dump() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Add a key/value pair or change a value.
   *  \deprecated Obsolete coding style.
   */
  template <class T>
  bool SetValue(const icl_core::String &key,
                typename icl_core::ConvertToRef<T>::ToConstRef value)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Register an observer which gets notified of changed key/value
   *  pairs.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RegisterObserver(ConfigObserver *observer, const String &key = "")
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Unregister an observer so it does not get notified of changes
   *  anymore.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void UnregisterObserver(ConfigObserver *observer)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  //! Creates an empty configuration object.
  ConfigManager();

  //! Reads configuration from a file.
  bool load(const icl_core::String& filename);

  //! Notify all observers about a changed key/value pair
  void notify(const icl_core::String &key) const;

  void readXml(const ::icl_core::String& prefix, TiXmlNode *node, FilePath fp, bool extend_prefix = true);
  void readAttributeTree(const icl_core::String& prefix, AttributeTree *at, bool extend_prefix = true);

  //typedef ::icl_core::Map< ::icl_core::String, ::icl_core::String> KeyValueMap;
  //KeyValueMap m_config_items;
  bool m_initialized;

  ConfigParameterList m_parameter_list;
  ConfigPositionalParameterList m_postional_parameter_list;

  typedef icl_core::Map<icl_core::String, icl_core::List<ConfigObserver*> > ObserverMap;
  ObserverMap m_observers;
};

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

template <class T> ICL_CORE_VC_DEPRECATE_STYLE
bool ConfigManager::SetValue(const icl_core::String &key,
                             typename icl_core::ConvertToRef<T>::ToConstRef value)
{
  return setValue<T>(key, value);
}

#endif
  /////////////////////////////////////////////////

}
}

#endif
