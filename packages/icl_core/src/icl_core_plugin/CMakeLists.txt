# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_plugin" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
    PluginManagerBase.cpp
    Logging.cpp
)

ICMAKER_ADD_HEADERS(
    icl_core_plugin.h
    ImportExport.h
    PluginManager.h
    PluginManagerBase.h
    Logging.h
    Plugin.h
)

IF(ICMAKER_DEPRECATED_STYLE)
ICMAKER_ADD_HEADERS(
    tPluginManager.h
    tPluginManagerBase.h
)
ENDIF()

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_PLUGIN_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_PLUGIN_)
ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  Boost_FILESYSTEM
  )

IF(WIN32)
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT DlfcnWin32)
ENDIF()

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core_logging
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(${icmaker_target})
