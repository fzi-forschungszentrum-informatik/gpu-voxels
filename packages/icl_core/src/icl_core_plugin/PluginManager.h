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
 * \author  Dennis Nienhueser <nienhues@fzi.de>
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2009-11-22
 *
 * \brief   Contains PluginManager
 *
 * \b icl_core::plugin::PluginManager
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_PLUGIN_PLUGIN_MANAGER_H_INCLUDED
#define ICL_CORE_PLUGIN_PLUGIN_MANAGER_H_INCLUDED

#include "icl_core_plugin/ImportExport.h"
#include "icl_core_plugin/Logging.h"
#include "icl_core_plugin/PluginManagerBase.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <cassert>
#include <string>
#include <list>
#include <map>
#include <iostream>
#include <sstream>

#include <dlfcn.h>

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
//! Plugin framework.
namespace plugin {

typedef void* PluginHandle;

/*! Singleton responsible for loading tControlWidget plugins.
 */
template <class T, const char* const plugin_dir>
class PluginManager: public PluginManagerBase
{
public:
  ~PluginManager()
  {
    while (!m_plugin_handles.empty() && !m_loaded_plugins.empty())
    {
      closePluginHandle(m_loaded_plugins.begin()->first);
    }

    static PluginManager<T, plugin_dir>* m_instance;
    m_instance = 0;
    (void) m_instance;
  }

  //! Singleton instance accessor.
  static PluginManager<T, plugin_dir> *instance()
  {
    //! Singleton instance.
    static PluginManager<T, plugin_dir> *m_instance = 0;

    if (!m_instance)
    {
      m_instance = new PluginManager<T, plugin_dir>();
    }

    assert(m_instance);
    return m_instance;
  }

  //! Initializes the PluginManager.
  bool initialize(bool load_lazy = true, bool recursive_search = false)
  {
    if (isInitialized() && loadLazy() && !load_lazy)
    {
      loadPlugins();
      return true;
    }

    if (isInitialized())
    {
      LOGGING_WARNING(Plugin, "The plugin framework is already initialized!" << icl_core::logging::endl);
      return true;
    }

    initializePlugins(recursive_search);

    if (!load_lazy)
    {
      loadPlugins();
    }
    else
    {
      m_lazy_loading = true;
    }

    return true;
  }

  bool isInitialized() const
  {
    return m_initialized;
  }

  bool loadLazy() const
  {
    return m_lazy_loading;
  }

  /*! \return All plugin identifiers currently available.
   */
  std::list<std::string> availablePlugins()
  {
    if (!isInitialized())
    {
      initialize(false);
    }

    if (loadLazy())
    {
      loadPlugins();
    }

    std::list<std::string> result;
    typename std::map<std::string, PluginHandle>::const_iterator iter;
    for (iter = m_plugin_handles.begin(); iter != m_plugin_handles.end(); ++iter)
    {
      result.push_back(iter->first);
    }

    typename std::map<std::string, T*>::const_iterator siter;
    for (siter = m_static_plugins.begin(); siter != m_static_plugins.end(); ++siter)
    {
      result.push_back(siter->first);
    }

    return result;
  }

  /*! \return All plugins currently instantiated.
   */
  std::list<T*> plugins()
  {
    if (!isInitialized())
    {
      initialize(false);
    }

    if (loadLazy())
    {
      loadPlugins();
    }

    std::list<T*> result;
    typename std::map<std::string, T*>::const_iterator iter;
    for (iter = m_loaded_plugins.begin(); iter != m_loaded_plugins.end(); ++iter)
    {
      result.push_back(iter->second);
    }

    typename std::map<std::string, T*>::const_iterator siter;
    for (siter = m_static_plugins.begin(); siter != m_static_plugins.end(); ++siter)
    {
      result.push_back(siter->second);
    }

    return result;
  }

  /*! Return the plugin with the given \a name.
   *  \param identifier Plugin name
   */
  T* plugin(const std::string &identifier)
  {
    if (!isInitialized())
    {
      initialize(false);
    }

    if (loadLazy())
    {
      loadPlugins();
    }

    if (m_static_plugins.find(identifier) != m_static_plugins.end())
    {
      return m_static_plugins[identifier];
    }

    return m_loaded_plugins[identifier];
  }

  /*! Create a new plugin instance.
   *  Warning: Manager doesn't care about instantiated object.
   */
  T* createPluginInstance(const std::string &identifier)
  {
    if (!isInitialized())
    {
      initialize(false);
    }

    if (loadLazy())
    {
      loadPlugins();
    }

    PluginHandle plugin_handle = m_plugin_handles[identifier];
    return loadPluginInstance(plugin_handle, identifier);
  }

  /*! Manually add an already created plugin instance.  Can be useful
   *  for example if the plugin is linked against the program already
   *  and does not need to be loaded dynamically. Ownership of the
   *  instance retains with the caller of the function.
   *  \see removeStaticPlugin
   *  \note Prior to deleting the plugin instance you must call
   *        removeStaticPlugin().
   */
  void addStaticPlugin(T* instance)
  {
    removeStaticPlugin(instance->Identifier());
    m_static_plugins[instance->Identifier()] = instance;
    LOGGING_DEBUG(Plugin, "Static plugin " << instance->Identifier() << " added." << icl_core::logging::endl);
  }

  /*! Remove a plugin that was added via addStaticPlugin.
   *  \see addStaticPlugin
   */
  void removeStaticPlugin(const std::string &identifier)
  {
    m_static_plugins.erase(identifier);
    LOGGING_DEBUG(Plugin, "Static plugin " << identifier << " removed."
                  << icl_core::logging::endl);
  }

  /*! Returns \c true if a plugin with the given identifier was added
   *  with addStaticPlugin() and not removed afterwards.
   *  \see addStaticPlugin removeStaticPlugin
   */
  bool isStaticPlugin(const std::string &identifier) const
  {
    return m_static_plugins.find(identifier) != m_static_plugins.end();
  }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Singleton instance accessor.
   *  \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE PluginManager<T, plugin_dir> *Instance()
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return instance(); }

  /*! Initializes the PluginManager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Initialize(bool load_lazy = true,
                                              bool recursive_search = false)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return initialize(load_lazy, recursive_search); }

  ICL_CORE_VC_DEPRECATE_STYLE bool IsInitialized() const
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return isInitialized(); }

  ICL_CORE_VC_DEPRECATE_STYLE bool LoadLazy() const
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return loadLazy(); }

  /*! \return All plugin identifiers currently available.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE std::list<std::string> AvailablePlugins()
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return availablePlugins(); }

  /*! \return All plugins currently instantiated.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE std::list<T*> Plugins()
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return plugins(); }

  /*! Return the plugin with the given \a name.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE T* Plugin(const std::string &identifier)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return plugin(identifier); }

  /*! Create a new plugin instance.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE T* CreatePluginInstance(const std::string &identifier)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return createPluginInstance(identifier); }

  /*! Manually add an already created plugin instance.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddStaticPlugin(T* instance)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { addStaticPlugin(instance); }

  /*! Remove a plugin that was added via AddStaticPlugin.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RemoveStaticPlugin(const std::string &identifier)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { removeStaticPlugin(identifier); }

  /*! Returns \c true if a plugin with the given identifier was added
   *  with AddStaticPlugin() and not removed afterwards.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsStaticPlugin(const std::string &identifier) const
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return isStaticPlugin(identifier); }

#endif
  /////////////////////////////////////////////////

protected:
  //! Protected default constructor.
  PluginManager()
    : PluginManagerBase(plugin_dir),
      m_initialized(false),
      m_lazy_loading(false)
  {
    // nothing to do
  }

  /*! Search all files in the specified directories and add plugins
   *  found.
   */
  void initializePlugins(bool recursive_search)
  {
    assert(!isInitialized());
    clearErrorMessages();
    m_initialized = true;

    StringList paths = pluginPaths();
    for (StringList::const_iterator iter = paths.begin(); iter != paths.end(); ++iter)
    {
      std::string path = *iter;
      boost::filesystem::path bpath(path);
      LOGGING_TRACE(Plugin, "Loading plugins from " << path << icl_core::logging::endl);

      bool found = false;
      try
      {
        found = exists(bpath);
      }
      catch (const boost::filesystem::filesystem_error &e)
      {
        LOGGING_DEBUG(Plugin, "Exception when examining directory " << path
                      << ": " << e.what() << icl_core::logging::endl);
        // Ignored
      }

      if (!found)
      {
        LOGGING_DEBUG(Plugin, "Ignoring non-existing plugin path " << path << icl_core::logging::endl);
        continue;
      }

      if (!recursive_search)
      {
        boost::filesystem::directory_iterator end_iter;
        for (boost::filesystem::directory_iterator dir_iter(bpath); dir_iter != end_iter; ++dir_iter)
        {
          if (boost::filesystem::is_regular_file(dir_iter->status()))
          {
#if BOOST_FILESYSTEM_VERSION == 2
            loadHandle(dir_iter->string());
#else
            loadHandle(dir_iter->path().string());
#endif
          }
        }
      }
      else
      {
        boost::filesystem::recursive_directory_iterator end_iter;
        for (boost::filesystem::recursive_directory_iterator dir_iter(bpath);
             dir_iter != end_iter; ++dir_iter)
        {
          if (boost::filesystem::is_regular_file(dir_iter->status()))
          {
#if BOOST_FILESYSTEM_VERSION == 2
            loadHandle(dir_iter->string());
#else
            loadHandle(dir_iter->path().string());
#endif
          }
        }
      }
    }

    LOGGING_DEBUG(Plugin, "Initialized " << m_plugin_handles.size() << " plugins."
                  << icl_core::logging::endl);
  }

  /*! Load the plugin handle.
   */
  void loadHandle(std::string plugin)
  {
#ifdef _WIN32
    if (boost::algorithm::ends_with(plugin, ".lib")) {
        LOGGING_DEBUG(Plugin, "Ignoring .lib file in plugin directory: " << plugin << icl_core::logging::endl);
        return;
    }
#endif

    PluginHandle plugin_handle = dlopen(plugin.c_str(), RTLD_LAZY);

    dlerror(); // Clear any previous errors.
    typename T::identifier* get_identifier
      = (typename T::identifier*) dlsym(plugin_handle, "identifier");

    typename T::basetype* get_basetype
      = (typename T::basetype*) dlsym(plugin_handle, "basetype");

    const char* dlsym_error = dlerror();
    if (dlsym_error)
    {
#ifdef _WIN32
      LPVOID msg_buffer;
      unsigned error_code = GetLastError();

      FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                    | FORMAT_MESSAGE_FROM_SYSTEM,
                    NULL,
                    error_code,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                    (LPTSTR) &msg_buffer,
                    0, NULL);

      std::ostringstream error;
      error << "Could not open dll: (" << error_code << ") " << msg_buffer;
      addErrorMessage(error.str());
      /*! \todo: use snprintf to print msg_buffer to the stream and re-enable next line */
      //LOGGING_ERROR(Plugin, error.str() << icl_core::logging::endl);
      printf("could not open dll: (%d) %s\n", error_code, msg_buffer);
      LocalFree(msg_buffer);
#else // _WIN32
      std::ostringstream error;
      error << "Cannot load plugin " << plugin << ": " << dlsym_error;
      addErrorMessage(error.str());
      LOGGING_ERROR(Plugin, error.str() << icl_core::logging::endl);
      // cannot find symbol
#endif // _WIN32
    }
    else
    {
      if (!get_identifier || !get_basetype)
      {
        std::ostringstream error;
        error << "Identifier or basetype method missing in plugin " << plugin;
        addErrorMessage(error.str());
        LOGGING_ERROR(Plugin, error.str() << icl_core::logging::endl);
      }
      else if (strcmp(get_basetype(), typeid(T).name()) != 0)
      {
        LOGGING_WARNING(Plugin, "Plugin type mismatch: Exptected " << typeid(T).name()
                        << ", got " << get_basetype() << icl_core::logging::endl);
      }
      else
      {
        m_plugin_handles[get_identifier()] = plugin_handle;
        LOGGING_DEBUG(Plugin, "Initialized plugin " << get_identifier() << " of basetype "
                      << get_basetype() << icl_core::logging::endl);
      }
    }
  }

  /*! Loads an instance for all available plugins.  Manager will take
   *  care of these instances.
   */
  void loadPlugins()
  {
    m_lazy_loading = false;

    typename std::map<std::string, PluginHandle>::const_iterator iter;
    for (iter = m_plugin_handles.begin(); iter != m_plugin_handles.end(); ++iter)
    {
      if (m_loaded_plugins.find(iter->first) == m_loaded_plugins.end())
      {
        m_loaded_plugins[iter->first] = createPluginInstance(iter->first);
      }
    }

    LOGGING_DEBUG(Plugin, "Loaded " << m_loaded_plugins.size() << " plugins." << icl_core::logging::endl);
  }

  /*! Unload a Plugin with a given identifier.
   */
  void unloadPlugin(const std::string &identifier)
  {
    delete m_loaded_plugins[identifier];
    m_loaded_plugins.erase(identifier);
  }

  /*! Closes a Plugin handle with a given identifier.
   */
  void closePluginHandle(const std::string &identifier)
  {
    PluginHandle plugin_handle = m_plugin_handles[identifier];
    if (plugin_handle)
    {
      LOGGING_DEBUG(Plugin, "Close plugin " << identifier << icl_core::logging::endl);

      if (m_loaded_plugins.find(identifier) != m_loaded_plugins.end())
      {
        T* plugin_instance = m_loaded_plugins[identifier];
        delete plugin_instance;
        m_loaded_plugins.erase(identifier);
      }

      dlclose(plugin_handle);
    }
    m_plugin_handles.erase(identifier);
  }

  /*! Creates a new plugin instance.
   */
  T* loadPluginInstance(PluginHandle plugin_handle, const std::string &identifier)
  {
    if (plugin_handle)
    {
      // clear errors
      dlerror();

      typename T::load_plugin* create_instance
        = (typename T::load_plugin*) dlsym(plugin_handle,
                                              "load_plugin");
      const char* dlsym_error = dlerror();
      if (dlsym_error)
      {
#ifdef _WIN32
        LPVOID msg_buffer;
        unsigned error_code = GetLastError();

        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                      | FORMAT_MESSAGE_FROM_SYSTEM,
                      NULL,
                      error_code,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &msg_buffer,
                      0, NULL);

        std::ostringstream error;
        error << "Could not open dll: (" << error_code << ") " << msg_buffer;
        addErrorMessage(error.str());
        LOGGING_ERROR(Plugin, error.str() << icl_core::logging::endl);
        LocalFree(msg_buffer);
#else // _WIN32
        std::ostringstream error;
        error << "Cannot load plugin " << identifier << ": " << dlsym_error;
        addErrorMessage(error.str());
        LOGGING_ERROR(Plugin, error.str() << icl_core::logging::endl);
        // cannot find symbol
#endif // _WIN32
      }
      else
      {
        T* plugin_instance = create_instance();
        if (!plugin_instance)
        {
          std::ostringstream error;
          error << "Cannot cast plugin " << identifier;
          addErrorMessage(error.str());
          LOGGING_ERROR(Plugin, error.str() << icl_core::logging::endl);
        }
        else
        {
          LOGGING_DEBUG(Plugin, "Loaded plugin " << plugin_instance->Identifier() << icl_core::logging::endl);
        }
        return plugin_instance;
      }
    }
    LOGGING_ERROR(Plugin, "No valid plugin handle available for " << identifier << icl_core::logging::endl);
    return NULL;
  }

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Search all files in the specified directories and add plugins
   *  found.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void InitializePlugins(bool recursive_search)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { initializePlugins(recursive_search); }

  /*! Load the plugin handle.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void LoadHandle(std::string plugin)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { loadHandle(plugin); }

  /*! Loads an instance for all available plugins.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void LoadPlugins()
    ICL_CORE_GCC_DEPRECATE_STYLE
  { loadPlugins(); }

  /*! Unload a Plugin with a given identifier.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void UnloadPlugin(const std::string &identifier)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { unloadPlugin(identifier); }

  /*! Closes a Plugin handle with a given identifier.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void ClosePluginHandle(const std::string &identifier)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { closePluginHandle(identifier); }

  /*! Creates a new plugin instance.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE T* LoadPluginInstance(PluginHandle plugin_handle, const std::string &identifier)
    ICL_CORE_GCC_DEPRECATE_STYLE
  { return loadPluginInstance(plugin_handle, identifier); }

#endif
  /////////////////////////////////////////////////

  //! Plugins loaded from disk via ltdl.
  std::map<std::string, T*> m_loaded_plugins;

  //! Loadable plugin handles.
  std::map<std::string, PluginHandle> m_plugin_handles;

  //! Plugins available from an already loaded lib, added manually.
  std::map<std::string, T*> m_static_plugins;

  bool m_initialized;

  bool m_lazy_loading;
};

}
}

#endif
