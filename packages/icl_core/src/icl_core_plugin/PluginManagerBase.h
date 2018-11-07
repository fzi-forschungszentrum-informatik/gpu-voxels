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
 * \author  Dennis Nienhüser <nienhues@fzi.de>
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2009-12-21
 *
 * \brief   Contains PluginManagerBase
 *
 * \b icl_core::plugin::PluginManagerBase
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_PLUGIN_PLUGIN_MANAGER_BASE_H_INCLUDED
#define ICL_CORE_PLUGIN_PLUGIN_MANAGER_BASE_H_INCLUDED

#include "icl_core_plugin/ImportExport.h"
#include "icl_core_plugin/Logging.h"

#include <string>
#include <list>

namespace icl_core {
namespace plugin {

typedef std::list<std::string> StringList;

class ICL_CORE_PLUGIN_IMPORT_EXPORT PluginManagerBase
{
public:
  /*! Constructor.
   *  \param plugin_dir The directory to search for plugins.
   */
  explicit PluginManagerBase(std::string plugin_dir = "");

  /*! \return A list of plugin search paths
   */
  StringList pluginPaths();

  /*! Add a path to the plugin search path
   *  \param path (Full) path to be added to the list of directories
   *         being searched for plugins
   */
  void addPluginPath(const std::string &path);

  /*! \return A list of error messages containing error descriptions
   *          for every plugin that failed to load.  The list will be
   *          empty if all plugins loaded successfully.
   */
  StringList errorMessages();

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! \return A list of plugin search paths
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE StringList PluginPaths() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Add a path to the plugin search path
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddPluginPath(const std::string &path)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! \return A list of error messages containing error descriptions
   *          for every plugin that failed to load.  The list will be
   *          empty if all plugins loaded successfully.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE StringList ErrorMessages() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

protected:
  void clearErrorMessages();

  void addErrorMessage(const std::string &message);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  ICL_CORE_VC_DEPRECATE_STYLE void ClearErrorMessages() ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE void AddErrorMessage(const std::string &message)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

  StringList m_error_messages;

private:
  StringList m_plugin_paths;
};

}
}

#endif
