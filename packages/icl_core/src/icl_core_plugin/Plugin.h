// ----------------------------------------------------------
/*
 * Plugin.h
 *
 * Created by schamm on 22.11.2009.
 * Copyright 2009
 * Company Forschungszentrum Informatik (FZI), Abteilung IDS.
 * All rights reserved.
 *
 */
// ----------------------------------------------------------
/*!
 *  \file       Plugin.h
 *  \author     <a href="mailto:schamm@fzi.de">Thomas Schamm</a>
 *  \date       22.11.2009
 *
 *  \brief      Base header file for the plugin framework.
 *
 */
// ----------------------------------------------------------


#ifndef ICL_CORE_PLUGIN_PLUGIN_H_INCLUDED
#define ICL_CORE_PLUGIN_PLUGIN_H_INCLUDED

#include "icl_core_plugin/ImportExport.h"
#include "icl_core_plugin/PluginManager.h"
#include "icl_core_plugin/Logging.h"

#include <string>

#if defined(_SYSTEM_WIN32_) && defined(_IC_STATIC_)
  #define ICL_CORE_PLUGIN_LINKAGE static
#else // defined(_SYSTEM_WIN32_) && defined(_IC_STATIC_)
  #define ICL_CORE_PLUGIN_LINKAGE
#endif // defined(_SYSTEM_WIN32_) && defined(_IC_STATIC_)

#define ICL_CORE_PLUGIN_DECLARE_PLUGIN_INTERFACE(name)  \
  public:                                               \
  virtual std::string Identifier() const = 0;           \
  typedef const char* identifier();                     \
  typedef const char* basetype();                       \
  typedef name* load_plugin();                          \
  typedef void unload_plugin(name*);

#define ICL_CORE_PLUGIN_DECLARE_PLUGIN          \
  public:                                       \
  virtual std::string Identifier() const;       \

#define ICL_CORE_PLUGIN_REGISTER_PLUGIN(base, derived, plugin_identifier) \
  std::string derived::Identifier() const                               \
  {                                                                     \
    return plugin_identifier;                                           \
  }                                                                     \
                                                                        \
  extern "C" ICL_CORE_PLUGIN_LINKAGE const char * identifier()          \
  {                                                                     \
    return plugin_identifier;                                           \
  }                                                                     \
                                                                        \
  extern "C" ICL_CORE_PLUGIN_LINKAGE const char * basetype()            \
  {                                                                     \
    return typeid(base).name();                                         \
  }                                                                     \
                                                                        \
  extern "C" ICL_CORE_PLUGIN_LINKAGE base* load_plugin()                \
  {                                                                     \
    return new derived;                                                 \
  }                                                                     \
                                                                        \
  extern "C" ICL_CORE_PLUGIN_LINKAGE void unload_plugin(derived* p)     \
  {                                                                     \
    delete p;                                                           \
  }

#define ICL_CORE_PLUGIN_MANAGER_DEFINITION(name, interface, directory)  \
  name : public icl_core::plugin::PluginManager<interface, directory> { \
    name()                                                              \
    {                                                                   \
      LOGGING_TRACE(icl_core::plugin::Plugin, "Using plugin search path " \
                    << directory << icl_core::logging::endl);           \
    }                                                                   \
  };

#define ICL_CORE_PLUGIN_DECLARE_PLUGIN_MANAGER(name, interface, directory) \
  class ICL_CORE_PLUGIN_MANAGER_DEFINITION(name, interface, directory)
#define ICL_CORE_PLUGIN_DECLARE_PLUGIN_MANAGER_IMPORT_EXPORT(decl, name, interface, directory) \
  class decl ICL_CORE_PLUGIN_MANAGER_DEFINITION(name, interface, directory)

#endif /* _icl_core_plugin_Plugin_h_ */
