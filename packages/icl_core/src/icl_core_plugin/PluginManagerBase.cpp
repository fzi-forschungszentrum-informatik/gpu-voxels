// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Dennis Nienhüser <nienhues@fzi.de>
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2009-12-21
 *
 */
//----------------------------------------------------------------------
#include "icl_core_plugin/PluginManagerBase.h"

namespace icl_core {
namespace plugin {

PluginManagerBase::PluginManagerBase(std::string plugin_dir)
{
  addPluginPath(plugin_dir);
}

StringList PluginManagerBase::pluginPaths()
{
  return m_plugin_paths;
}

void PluginManagerBase::addPluginPath(const std::string &path)
{
  if (path != "")
  {
    m_plugin_paths.push_back(path);
  }
  else
  {
    LOGGING_TRACE(Plugin, "Cannot use an empty plugin path" << icl_core::logging::endl);
  }
}

StringList PluginManagerBase::errorMessages()
{
  return m_error_messages;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! \return A list of plugin search paths
 *  \deprecated Obsolete coding style.
 */
StringList PluginManagerBase::PluginPaths()
{
  return pluginPaths();
}

/*! Add a path to the plugin search path
 *  \deprecated Obsolete coding style.
 */
void PluginManagerBase::AddPluginPath(const std::string &path)
{
  addPluginPath(path);
}

/*! \return A list of error messages containing error descriptions
 *          for every plugin that failed to load.  The list will be
 *          empty if all plugins loaded successfully.
 *  \deprecated Obsolete coding style.
 */
StringList PluginManagerBase::ErrorMessages()
{
  return errorMessages();
}

#endif
/////////////////////////////////////////////////

void PluginManagerBase::clearErrorMessages()
{
  m_error_messages.clear();
}

void PluginManagerBase::addErrorMessage(const std::string &message)
{
  m_error_messages.push_back(message);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

void PluginManagerBase::ClearErrorMessages()
{
  clearErrorMessages();
}

void PluginManagerBase::AddErrorMessage(const std::string &message)
{
  addErrorMessage(message);
}

#endif
/////////////////////////////////////////////////

}
}
