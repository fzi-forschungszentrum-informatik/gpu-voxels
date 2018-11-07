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
 * \date    2007-05-12
 */
//----------------------------------------------------------------------
#include "icl_core_config/ConfigManager.h"

#include <assert.h>
#include <iostream>
#include <tinyxml.h>

#include "icl_core/KeyValueDirectory.hpp"
#include "icl_core_config/AttributeTree.h"
#include "icl_core_config/Config.h"
#include "icl_core_config/ConfigObserver.h"
#include "icl_core_config/GetoptParser.h"

namespace icl_core {

// Explicit template instantiation!
template class KeyValueDirectory<String>;

namespace config {

ConfigManager& ConfigManager::instance()
{
  static ConfigManager instance;
  return instance;
}

void ConfigManager::addParameter(const ConfigParameter& parameter)
{
  // Add to the own parameter list.
  if (parameter.configKey() != "")
  {
    m_parameter_list.push_back(parameter);
  }

  // Delegate to Getopt.
  Getopt::instance().addParameter(parameter);
}

void ConfigManager::addParameter(const ConfigParameterList& parameters)
{
  for (ConfigParameterList::const_iterator it = parameters.begin(); it != parameters.end(); ++it)
  {
    addParameter(*it);
  }
}

void ConfigManager::addParameter(const ConfigPositionalParameter &parameter)
{
  // Add to the own parameter list.
  if (parameter.configKey() != "")
  {
    m_postional_parameter_list.push_back(parameter);
  }

  // Delegate to Getopt.
  Getopt::instance().addParameter(parameter);
}

void ConfigManager::addParameter(const ConfigPositionalParameterList &parameters)
{
  for (ConfigPositionalParameterList::const_iterator it = parameters.begin(); it != parameters.end(); ++it)
  {
    addParameter(*it);
  }
}

bool ConfigManager::initialize()
{
  if (isInitialized())
  {
    std::cerr << "CONFIG WARNING: The configuration framework is already initialized!" << std::endl;
    return true;
  }

  if (Getopt::instance().paramOptPresent("configfile"))
  {
    // Read the configuration file.
    icl_core::String filename = Getopt::instance().paramOpt("configfile");
    if (!load(filename))
    {
      std::cerr << "CONFIG ERROR: The configuration file '" << filename << "' could not be loaded!"
          << std::endl;
      return false;
    }
    insert(CONFIGFILE_CONFIG_KEY, filename);
    notify(CONFIGFILE_CONFIG_KEY);
  }

  // Check for registered parameters.
  for (ConfigParameterList::const_iterator it = m_parameter_list.begin(); it != m_parameter_list.end(); ++it)
  {
    if (it->configKey() != "")
    {
      // Fill the configuration parameter from the commandline.
      if (Getopt::instance().paramOptPresent(it->option()))
      {
        insert(it->configKey(), Getopt::instance().paramOpt(it->option()));
        notify(it->configKey());
      }
      // If the parameter is still not present but has a default value, then set it.
      else if (!hasKey(it->configKey()) && it->hasDefaultValue())
      {
          insert(it->configKey(), it->defaultValue());
          notify(it->configKey());
      }
    }
  }

  // Check for registered positional parameters.
  for (ConfigPositionalParameterList::const_iterator it = m_postional_parameter_list.begin(); it != m_postional_parameter_list.end(); ++it)
  {
    if (it->configKey() != "")
    {
      // Fill the configuration parameter from the commandline.
      if (Getopt::instance().paramOptPresent(it->name()))
      {
        insert(it->configKey(), Getopt::instance().paramOpt(it->name()));
        notify(it->configKey());
      }
      // If the parameter is still not present but has a default value, then set it.
      else if (!hasKey(it->configKey()) && it->hasDefaultValue())
      {
          insert(it->configKey(), it->defaultValue());
          notify(it->configKey());
      }
    }
  }

  // Check for option parameters.
  Getopt::KeyValueList option_params = Getopt::instance().paramPrefixOpt("config-option");
  for (Getopt::KeyValueList::const_iterator it = option_params.begin(); it != option_params.end(); ++it)
  {
    insert(it->m_key, it->m_value);
    notify(it->m_key);
  }

  // Optionally dump the configuration.
  if (Getopt::instance().paramOptPresent("dump-config"))
  {
    dump();
  }

  m_initialized = true;
  return true;
}

void ConfigManager::dump() const
{
  std::cout << "--- BEGIN CONFIGURATION DUMP ---" << std::endl;
  ConfigIterator it = find(".*");
  while (it.next())
  {
    std::cout << it.key() << " = '" << it.value() << "'" << std::endl;
  }
  std::cout << "--- END CONFIGURATION DUMP ---" << std::endl;
}

ConfigManager::ConfigManager()
  : m_initialized(false)
{
  addParameter(ConfigParameter("configfile:", "c", CONFIGFILE_CONFIG_KEY,
                               "Specifies the path to the configuration file."));
  Getopt::instance().addParameter(GetoptParameter("dump-config", "dc",
                                  "Dump the configuration read from the configuration file."));
  Getopt::instance().addParameter(GetoptParameter("config-option:", "o",
                                                    "Overwrite a configuration option.", true));
}

bool ConfigManager::load(const icl_core::String& filename)
{
  FilePath fp(filename.c_str());

  if (fp.extension() == ".AttributeTree" || fp.extension() == ".tree")
  {
    AttributeTree attribute_tree;
    int res = attribute_tree.load(filename.c_str());
    if (res != AttributeTree::eFILE_LOAD_ERROR)
    {
      if (res == AttributeTree::eOK)
      {
        readAttributeTree("", attribute_tree.root(), false);
      }
      return true;
    }
    else
    {
      std::cerr << "CONFIG ERROR: Could not load configuration file '" << filename << std::endl;
      return false;
    }
  }
  else
  {
    TiXmlDocument doc(filename.c_str());
    if (doc.LoadFile())
    {
      TiXmlElement *root_element = doc.RootElement();
      if (root_element != 0)
      {
        readXml("", root_element, fp, false);
      }
      return true;
    }
    else
    {
      std::cerr << "CONFIG ERROR: Could not load configuration file '" << filename << "' (" << doc.ErrorRow()
          << ", " << doc.ErrorCol() << "): " << doc.ErrorDesc() << std::endl;
      return false;
    }
  }
}

void ConfigManager::readXml(const icl_core::String& prefix, TiXmlNode *node, FilePath fp, bool extend_prefix)
{
  icl_core::String node_name(node->Value());
  icl_core::String fq_node_name = prefix;
  if (extend_prefix)
  {
    fq_node_name = prefix + "/" + node_name;
  }

  TiXmlNode *child = node->IterateChildren(NULL);
  while (child != 0)
  {
    if (child->Type() == TiXmlNode::TINYXML_ELEMENT)
    {
      if (strcmp(child->Value(), "INCLUDE") == 0)
      {
        TiXmlElement *child_element = dynamic_cast<TiXmlElement*>(child);
        assert(child_element != NULL);
        const char *included_file = child_element->GetText();
        if (included_file != NULL)
        {
          load(fp.path() + included_file);
        }
      }
      else
      {
        readXml(fq_node_name, child, fp);
      }
    }
    else if (child->Type() == TiXmlNode::TINYXML_TEXT)
    {
      insert(fq_node_name, child->Value());
      notify(fq_node_name);
    }

    child = node->IterateChildren(child);
  }
}

void ConfigManager::readAttributeTree(const icl_core::String& prefix, AttributeTree *at, bool extend_prefix)
{
  icl_core::String node_name = "";
  if (at->getDescription() != NULL)
  {
    node_name = at->getDescription();
  }
  icl_core::String fq_node_name = prefix;
  if (extend_prefix)
  {
    fq_node_name = prefix + "/" + node_name;
  }

  if (!at->isComment() && at->attribute() != NULL)
  {
    insert(fq_node_name, at->attribute());
    notify(fq_node_name);
  }

  AttributeTree *child = at->firstSubTree();
  while (child != NULL)
  {
    readAttributeTree(fq_node_name, child);
    child = at->nextSubTree(child);
  }
}

void ConfigManager::registerObserver(ConfigObserver *observer, const icl_core::String &key)
{
  assert(observer && "Null must not be passed as config observer");

  m_observers[key].push_back(observer);

  if (key == "")
  {
    ConfigIterator iter = icl_core::config::ConfigManager::instance().find(".*");
    while (iter.next())
    {
      observer->valueChanged(iter.key());
    }
  }
  else if (find(key).next())
  {
    observer->valueChanged(key);
  }
}

void ConfigManager::unregisterObserver(ConfigObserver *observer)
{
  assert(observer && "Null must not be passed as config observer");

  icl_core::Map<icl_core::String, icl_core::List<ConfigObserver*> >::iterator iter;
  for (iter = m_observers.begin(); iter != m_observers.end(); ++iter)
  {
    iter->second.remove(observer);
  }
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Get the singleton ConfigManager instance.
 *  \deprecated Obsolete coding style.
 */
ConfigManager& ConfigManager::Instance()
{
  return instance();
}

/*! Adds a commandline parameter.
 *  \deprecated Obsolete coding style.
 */
void ConfigManager::AddParameter(const ConfigParameter& parameter)
{
  addParameter(parameter);
}

/*! Adds a list of commandline parameters.
 *  \deprecated Obsolete coding style.
 */
void ConfigManager::AddParameter(const ConfigParameterList& parameters)
{
  addParameter(parameters);
}

/*! Initializes ConfigManager.
 *  \deprecated Obsolete coding style.
 */
bool ConfigManager::Initialize()
{
  return initialize();
}

/*! Check if the configuration framework has already been
 *  initialized.
 *  \deprecated Obsolete coding style.
 */
bool ConfigManager::IsInitialized() const
{
  return isInitialized();
}

/*! Dumps all configuration keys and the corresponding values to
 *  stdout.
 *  \deprecated Obsolete coding style.
 */
void ConfigManager::Dump() const
{
  dump();
}

/*! Register an observer which gets notified of changed key/value
 *  pairs.
 *  \deprecated Obsolete coding style.
 */
void ConfigManager::RegisterObserver(ConfigObserver *observer, const String &key)
{
  registerObserver(observer, key);
}

/*! Unregister an observer so it does not get notified of changes
 *  anymore.
 *  \deprecated Obsolete coding style.
 */
void ConfigManager::UnregisterObserver(ConfigObserver *observer)

{
  unregisterObserver(observer);
}

#endif
/////////////////////////////////////////////////


void ConfigManager::notify(const icl_core::String &key) const
{
  icl_core::List<ConfigObserver*> observers;
  ObserverMap::const_iterator find_it = m_observers.find(key);
  if (find_it != m_observers.end())
  {
    observers.insert(observers.end(), find_it->second.begin(), find_it->second.end());
  }
  find_it = m_observers.find("");
  if (find_it != m_observers.end())
  {
    observers.insert(observers.end(), find_it->second.begin(), find_it->second.end());
  }

  icl_core::List<ConfigObserver*>::iterator iter;
  for (iter = observers.begin(); iter != observers.end(); ++iter)
  {
    (*iter)->valueChanged(key);
  }
}

}
}
