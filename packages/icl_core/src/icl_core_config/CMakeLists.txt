# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_config" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  AttributeTree.cpp
  Config.cpp
  ConfigManager.cpp
  ConfigIterator.cpp
  ConfigParameter.cpp
  ConfigPositionalParameter.cpp
  GetoptParameter.cpp
  GetoptPositionalParameter.cpp
  GetoptParser.cpp
  )

ICMAKER_ADD_HEADERS(
  AttributeTree.h
  Config.h
  ConfigEnum.h
  ConfigEnumDefault.h
  ConfigHelper.h
  ConfigIterator.h
  ConfigList.h
  ConfigManager.h
  ConfigObserver.h
  ConfigParameter.h
  ConfigPositionalParameter.h
  ConfigValue.h
  ConfigValueDefault.h
  ConfigValueIface.h
  ConfigValues.h
  GetoptParameter.h
  GetoptPositionalParameter.h
  GetoptParser.h
  icl_core_config.h
  ImportExport.h
  MemberEnum.h
  MemberValue.h
  MemberValueIface.h
  Util.h
  )

IF(ICMAKER_DEPRECATED_STYLE)
ICMAKER_ADD_HEADERS(
  tConfig.h
  tConfigIterator.h
  tConfigObserver.h
  tConfigParameter.h
  tConfigValues.h
  tGetopt.h
  tGetoptParameter.h
)
ENDIF()

IF(UNIX)
  # TODO: make this a FindFoo.cmake
  ICMAKER_LOCAL_CPPDEFINES(-D_IC_BUILDER_HAS_TIME_H_ -D_IC_BUILDER_HAS_SYS_TIME_H_)
ENDIF()

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_CONFIG_EXPORT_SYMBOLS)

ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_CONFIG_)

ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  Boost_REGEX
)

ICMAKER_EXTERNAL_DEPENDENCIES(OPTIONAL EXPORT
  OpenSpliceDDS
)

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
)

ICMAKER_INTERNAL_DEPENDENCIES( EXPORT
  tinyxml
)

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS("icl_core_config")
