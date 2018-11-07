# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# Copyright (c) 2018, FZI Forschungszentrum Informatik
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- END LICENSE BLOCK ------------------------------------------------

# ===================================================
# Macros that checks if module have been installed.
# After it adds module to build and define
# constants passed as second arg
# ===================================================

# ----------------------------------------------------------------------------
#                       MACROS USED INTERNALLY:
# ----------------------------------------------------------------------------
MACRO(PARSE_ARGUMENTS prefix arg_names option_names)
  SET(DEFAULT_ARGS)
  FOREACH(arg_name ${arg_names})
    SET(${prefix}_${arg_name})
  ENDFOREACH(arg_name)
  FOREACH(option ${option_names})
    SET(${prefix}_${option} FALSE)
  ENDFOREACH(option)

  SET(current_arg_name DEFAULT_ARGS)
  SET(current_arg_list)
  FOREACH(arg ${ARGN})
    SET(larg_names ${arg_names})
    LIST(FIND larg_names "${arg}" is_arg_name)
    IF (is_arg_name GREATER -1)
      SET(${prefix}_${current_arg_name} ${current_arg_list})
      SET(current_arg_name ${arg})
      SET(current_arg_list)
    ELSE (is_arg_name GREATER -1)
      SET(loption_names ${option_names})
      LIST(FIND loption_names "${arg}" is_option)
      IF (is_option GREATER -1)
             SET(${prefix}_${arg} TRUE)
      ELSE (is_option GREATER -1)
             SET(current_arg_list ${current_arg_list} ${arg})
      ENDIF (is_option GREATER -1)
    ENDIF (is_arg_name GREATER -1)
  ENDFOREACH(arg)
  SET(${prefix}_${current_arg_name} ${current_arg_list})
ENDMACRO(PARSE_ARGUMENTS)

MACRO(CAR var)
  SET(${var} ${ARGV1})
ENDMACRO(CAR)

MACRO(CDR var junk)
  SET(${var} ${ARGN})
ENDMACRO(CDR)

# ===================================================
# IcMaker macros:
# ===================================================


# ----------------------------------------------------------------------------
#                       DEFINE THE PACKAGE:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_REGISTER_PACKAGE _package)
  SET(icmaker_package "${_package}")
  SET(${icmaker_package}_COMPONENTS "" CACHE INTERNAL "")
  SET(${icmaker_package}_DEFINITIONS "" CACHE INTERNAL "")
ENDMACRO()

# ----------------------------------------------------------------------------
#                       DEFINE THE PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_SET _target)
  SET(icmaker_target "${_target}")

  PARSE_ARGUMENTS(ICMAKER_SET_OPT "IDE_FOLDER" "" ${ARGN})
  IF (ICMAKER_SET_OPT_IDE_FOLDER)
    SET(${icmaker_target}_IDE_FOLDER ${ICMAKER_SET_OPT_IDE_FOLDER})
  ENDIF ()

  IF(ICMAKER_VERBOSE)
    message(STATUS "================================")
    message(STATUS "I. Defined target ${icmaker_target}")
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS SOURCES TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_SOURCES)
  PARSE_ARGUMENTS(ADD_SOURCES "" "" ${ARGN})
  CAR(__sources "${ADD_SOURCES_DEFAULT_ARGS}")
  LIST(APPEND ${icmaker_target}_SOURCES ${__sources})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS HEADERS TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_HEADERS)
  PARSE_ARGUMENTS(ADD_HEADERS "" "" ${ARGN})
  CAR(__headers "${ADD_HEADERS_DEFAULT_ARGS}")
  LIST(APPEND ${icmaker_target}_HEADERS ${__headers})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS SWIG FILE TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_SWIG_FILE _file)
  SET(${icmaker_target}_SWIG_FILE "${_file}")
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS MOC HEADERS TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_MOC_HEADERS)
  IF (QT4_FOUND)
    PARSE_ARGUMENTS(ADD_MOCS "" "" ${ARGN})
    CAR(__headers "${ADD_MOCS_DEFAULT_ARGS}")
    LIST(APPEND ${icmaker_target}_HEADERS ${__headers})
    SET(__moc_sources)
    QT4_WRAP_CPP(__moc_sources ${__headers})
    LIST(APPEND ${icmaker_target}_GENERATED_SOURCES ${__moc_sources})
  ELSEIF (QT_FOUND)
    PARSE_ARGUMENTS(ADD_MOCS "" "" ${ARGN})
    CAR(__headers "${ADD_MOCS_DEFAULT_ARGS}")
    LIST(APPEND ${icmaker_target}_HEADERS ${__headers})
    SET(__moc_sources)
    QT_WRAP_CPP(qtwrapping __moc_sources ${__headers})
    LIST(APPEND ${icmaker_target}_GENERATED_SOURCES ${__moc_sources})
  ENDIF ()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS QT UI FILES TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_QT_UI_FILES)
  IF (QT4_FOUND)
    PARSE_ARGUMENTS(ADD_UIS "" "" ${ARGN})
    CAR(__ui_files "${ADD_UIS_DEFAULT_ARGS}")
    LIST(APPEND ${icmaker_target}_QT_UI_FILES ${__ui_files})
    QT4_WRAP_UI(__ui_headers ${__ui_files})
    LIST(APPEND ${icmaker_target}_GENERATED_HEADERS ${__ui_headers})
  ENDIF (QT4_FOUND)
  IF (QT5_FOUND)
    PARSE_ARGUMENTS(ADD_UIS "" "" ${ARGN})
    CAR(__ui_files "${ADD_UIS_DEFAULT_ARGS}")
    LIST(APPEND ${icmaker_target}_QT_UI_FILES ${__ui_files})
    QT5_WRAP_UI(__ui_headers ${__ui_files})
    LIST(APPEND ${icmaker_target}_GENERATED_HEADERS ${__ui_headers})
  ENDIF (QT5_FOUND)
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS QT RESOURCE FILES TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_QT_RESOURCE_FILES)
  IF (QT4_FOUND)
    PARSE_ARGUMENTS(ADD_RESOURCES "" "" ${ARGN})
    CAR(__resource_files "${ADD_RESOURCES_DEFAULT_ARGS}")
    LIST(APPEND ${icmaker_target}_QT_RESOURCE_FILES ${__resource_files})
    QT4_ADD_RESOURCES(__resource_sources ${__resource_files})
    LIST(APPEND ${icmaker_target}_GENERATED_SOURCES ${__resource_sources})
  ENDIF (QT4_FOUND)
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS ARBITRARY FILES TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_RESOURCES name)
  ADD_CUSTOM_TARGET(${icmaker_target}_${_name} SOURCES ${ARGN})
  SOURCE_GROUP(${name} FILES ${__resource_files})
ENDMACRO()


# ----------------------------------------------------------------------------
#                       LOCAL CPPDEFINES FOR PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_LOCAL_CPPDEFINES)
  PARSE_ARGUMENTS(LOCAL_CPPDEFINES "" "" ${ARGN})
  CAR(__cppdefines "${LOCAL_CPPDEFINES_DEFAULT_ARGS}")
  list(APPEND ${icmaker_target}_MACRO_DEFINITIONS ${__cppdefines})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       GLOBAL CPPDEFINES FOR PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_GLOBAL_CPPDEFINES)
  PARSE_ARGUMENTS(GLOBAL_CPPDEFINES "" "" ${ARGN})
  CAR(__cppdefines "${GLOBAL_CPPDEFINES_DEFAULT_ARGS}")
  list(APPEND ${icmaker_target}_MACRO_EXPORT_DEFINITIONS ${__cppdefines})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       INCLUDE DIRECTORIES FOR PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_INCLUDE_DIRECTORIES)
  PARSE_ARGUMENTS(INCLUDES "" "OPTIONAL" ${ARGN})
  CAR(__includes "${INCLUDES_DEFAULT_ARGS}")
  list(APPEND ${icmaker_target}_MACRO_EXPORT_INCLUDE_DIRS ${__includes})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       IDL DIRECTORIES FOR PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_IDL_DIRECTORIES)
  PARSE_ARGUMENTS(IDLDIRS "" "OPTIONAL" ${ARGN})
  CAR(__idldirs "${IDLDIRS_DEFAULT_ARGS}")
  list(APPEND ${icmaker_target}_MACRO_EXPORT_IDL_DIRS ${__idldirs})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       ADDS CUDA FILES TO PROJECT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_CUDA_FILES)
  IF (CUDA_FOUND)
    PARSE_ARGUMENTS(ADD_CUDA_FILES "" "" ${ARGN})
    CAR(__cuda_files "${ADD_CUDA_FILES_DEFAULT_ARGS}")
    LIST(APPEND ${icmaker_target}_CUDA_FILES ${__cuda_files})
#    SET(__cuda_generated_files)
#    CUDA_WRAP_SRCS(${icmaker_target}_CUDA_TARGET OBJ __cuda_generated_files ${__cuda_files} OPTIONS ${ICMAKER_CUDA_CPPDEFINES})
#    LIST(APPEND ${icmaker_target}_GENERATED_CUDA_OBJS ${__cuda_generated_files})
  ENDIF ()
ENDMACRO()

# ----------------------------------------------------------------------------
#           ADDS ROS SUBDIRECTORIES TO PROJECT IF ROS IS FOUND
# ----------------------------------------------------------------------------
MACRO(ICMAKER_ADD_ROS_SUBDIRECTORY)
  FIND_PACKAGE(ROS)   # \todo This should be placed somewhere more central!
  IF(ROS_FOUND)
    ADD_SUBDIRECTORY(${ARGN})
  ELSE()
    MESSAGE(STATUS "Warning: Omitting ROS subdirectory \"${ARGN}\" because ROS has not been found on the system! See http://wiki.ros.org/ROS/Installation if you want to use these features.")
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#   Add include dirs, definitions and link libraries from dependency:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_DEPENDECY_LIBS_AND_FLAGS)
  PARSE_ARGUMENTS(DEPENDECY_LIBS_AND_FLAGS
    ""
    ""
    ${ARGN}
    )
  CAR(__dependency "${DEPENDECY_LIBS_AND_FLAGS_DEFAULT_ARGS}")

  FOREACH(_dependency ${__dependency})
    SET(_name ${_dependency})

    IF(ICMAKER_VERBOSE)
      MESSAGE(STATUS "  Adding libs and flags of dependency ${_dependency}.")
    ENDIF()

    string(TOUPPER ${_name} _nameUpper)
    IF(${_nameUpper}_FOUND OR ${_name}_FOUND)

      SET(${_name}_PREFLIGHT_DEFINITIONS)
      SET(${_name}_PREFLIGHT_INCLUDE_DIRS)
      SET(${_name}_PREFLIGHT_IDL_DIRS)
      SET(${_name}_PREFLIGHT_LIBRARIES)
      SET(${_name}_PREFLIGHT_LDFLAGS)

      # Cache definitions and add _IC_BUILDER_-definition into ${_name}_PREFLIGHT_DEFINITIONS
      # Definitions may be added twice, however duplicates are removed afterwards
      list(APPEND ${_name}_PREFLIGHT_DEFINITIONS -D_IC_BUILDER_${_nameUpper}_ ${${_name}_DEFINITIONS} ${${_nameUpper}_DEFINITIONS})
      IF(${_name}_PREFLIGHT_DEFINITIONS)
        list(REMOVE_DUPLICATES ${_name}_PREFLIGHT_DEFINITIONS)
      ENDIF()

      # Cache possible include directories into ${_name}_PREFLIGHT_INCLUDE_DIRS
      IF(${_name}_INCLUDE_DIRS)
        list(APPEND ${_name}_PREFLIGHT_INCLUDE_DIRS ${${_name}_INCLUDE_DIRS})
      ENDIF()
      IF(${_name}_INCLUDE_DIR)
        list(APPEND ${_name}_PREFLIGHT_INCLUDE_DIRS ${${_name}_INCLUDE_DIR})
      ENDIF()
      IF(${_nameUpper}_INCLUDE_DIR AND NOT (${_nameUpper} STREQUAL ${_name}))
        list(APPEND ${_name}_PREFLIGHT_INCLUDE_DIRS ${${_nameUpper}_INCLUDE_DIR})
      ENDIF()
      IF(${_nameUpper}_INCLUDE_DIRS AND NOT (${_nameUpper} STREQUAL ${_name}))
        list(APPEND ${_name}_PREFLIGHT_INCLUDE_DIRS ${${_nameUpper}_INCLUDE_DIRS})
      ENDIF()
      IF(${_name}_PREFLIGHT_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES ${_name}_PREFLIGHT_INCLUDE_DIRS)
      ENDIF()
      
      # Cache possible idl directories into ${_name}_PREFLIGHT_IDL_DIRS
      IF(${_nameUpper}_IDL_DIRS)
        list(APPEND ${icmaker_target}_PREFLIGHT_IDL_DIRS ${${_nameUpper}_IDL_DIRS})
      ENDIF()
      
      # Cache possible libraries into ${_name}_PREFLIGHT_LIBRARIES
      IF(${_name}_LDFLAGS)
        list(APPEND ${_name}_PREFLIGHT_LDFLAGS ${${_name}_LDFLAGS})
      ENDIF()
      IF(${_name}_LIBS)
        list(APPEND ${_name}_PREFLIGHT_LIBRARIES ${${_name}_LIBS})
      ENDIF()
      IF(${_name}_LIBRARIES)
        IF(NOT ${_name} STREQUAL "Boost") # Do not add Boost_LIBRARIES
          list(APPEND ${_name}_PREFLIGHT_LIBRARIES ${${_name}_LIBRARIES})
        ENDIF()
      ENDIF()
      IF (${_nameUpper}_LIBRARIES AND NOT (${_nameUpper} STREQUAL ${_name}))
        list(APPEND ${_name}_PREFLIGHT_LIBRARIES ${${_nameUpper}_LIBRARIES})
      ENDIF()
      
      IF(ICMAKER_VERBOSE)
        MESSAGE(STATUS "    Definitions: ${${_name}_PREFLIGHT_DEFINITIONS}")
        MESSAGE(STATUS "    Include dirs: ${${_name}_PREFLIGHT_INCLUDE_DIRS}")
        MESSAGE(STATUS "    IDL dirs: ${${_name}_PREFLIGHT_IDL_DIRS}")
        MESSAGE(STATUS "    Libraries: ${${_name}_PREFLIGHT_LIBRARIES}")
        MESSAGE(STATUS "    LDFLAGS: ${${_name}_PREFLIGHT_LDFLAGS}")
        MESSAGE(STATUS "")
      ENDIF()
      
      IF(${_name}_PREFLIGHT_DEFINITIONS)
        list(APPEND ${icmaker_target}_MACRO_DEFINITIONS ${${_name}_PREFLIGHT_DEFINITIONS})
      ENDIF()
      IF(${_name}_PREFLIGHT_INCLUDE_DIRS)
        list(APPEND ${icmaker_target}_MACRO_INCLUDE_DIRS ${${_name}_PREFLIGHT_INCLUDE_DIRS})
      ENDIF()
      IF(${_name}_PREFLIGHT_IDL_DIRS)
        list(APPEND ${icmaker_target}_MACRO_IDL_DIRS ${${_name}_PREFLIGHT_IDL_DIRS})
      ENDIF()
      IF(${_name}_PREFLIGHT_LIBRARIES)
        list(APPEND ${icmaker_target}_MACRO_TARGET_LINK_LIBRARIES ${${_name}_PREFLIGHT_LIBRARIES})
      ENDIF()
      IF(${_name}_PREFLIGHT_LIBRARIES)
        list(APPEND ${icmaker_target}_MACRO_LDFLAGS ${${_name}_PREFLIGHT_LDFLAGS})
      ENDIF()
    ENDIF()
  ENDFOREACH(_dependency)
  
  IF(${icmaker_target}_MACRO_DEFINITIONS)
    LIST(REMOVE_DUPLICATES ${icmaker_target}_MACRO_DEFINITIONS)
  ENDIF()
  IF(${icmaker_target}_MACRO_INCLUDE_DIRS)
    LIST(REMOVE_DUPLICATES ${icmaker_target}_MACRO_INCLUDE_DIRS)
  ENDIF()
  IF(${icmaker_target}_MACRO_IDL_DIRS)
    LIST(REMOVE_DUPLICATES ${icmaker_target}_MACRO_IDL_DIRS)
  ENDIF()
  IF(${icmaker_target}_MACRO_LDFLAGS)
    LIST(REMOVE_DUPLICATES ${icmaker_target}_MACRO_LDFLAGS)
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       SCAN DEPENDENCIES RECURSIVELY:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_RECURSIVE_DEPENDENCIES _name)
  IF(${_name}_MACRO_EXPORT_DEPENDENCIES_C)
    FOREACH(_dependency ${${_name}_MACRO_EXPORT_DEPENDENCIES_C})
      LIST(FIND ${icmaker_target}_MACRO_DEPENDENCIES ${_dependency} SKIP_DEPENDENCY)
      IF(SKIP_DEPENDENCY EQUAL -1)
        ICMAKER_RECURSIVE_DEPENDENCIES(${_dependency})
      ENDIF()
    ENDFOREACH()
    list(APPEND ${icmaker_target}_MACRO_DEPENDENCIES ${${_name}_MACRO_EXPORT_DEPENDENCIES_C})
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       INTERNAL PROJECT DEPENDENCIES:
#
#   Deprecated.
#
# ----------------------------------------------------------------------------
MACRO(ICMAKER_INTERNAL_DEPENDENCIES)
  IF(ICMAKER_VERBOSE)
    message(STATUS "  Deprecated usage of ICMAKER_INTERNAL_DEPENDENCIES in: ${CMAKE_CURRENT_LIST_FILE}. Use ICMAKER_DEPENDENCIES instead!")
  ENDIF()
  ICMAKER_DEPENDENCIES(${ARGN})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       EXTERNAL PROJECT DEPENDENCIES:
#
#   Deprecated.
#
# ----------------------------------------------------------------------------
MACRO(ICMAKER_EXTERNAL_DEPENDENCIES)
  IF(ICMAKER_VERBOSE)
    message(STATUS "  Deprecated usage of ICMAKER_EXTERNAL_DEPENDENCIES in: ${CMAKE_CURRENT_LIST_FILE}. Use ICMAKER_DEPENDENCIES instead!")
  ENDIF()
  ICMAKER_DEPENDENCIES(${ARGN})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       PROJECT DEPENDENCIES:
#
#   Fill two lists: 
#     * ${icmaker_target}_MACRO_DEPENDENCIES: Dependencies used by target
#     * ${icmaker_target}_MACRO_EXPORT_DEPENDENCIES: Dependencies used by target and its parents
#
#   Each dependency listed is scanned recursively, if is has the exported deps. If so, these will
#   be used by the target. If a dependecy is optional and not available, the target will not fail
#   to be build. If optional dependencies are exported, they will will be exported like necessary
#   dependencies, as soon as they are available.
# ----------------------------------------------------------------------------
MACRO(ICMAKER_DEPENDENCIES)
  PARSE_ARGUMENTS(DEPENDENCY
    ""
    "OPTIONAL;EXPORT;FIND_PACKAGE"
    ${ARGN}
    )
  CAR(__dependency "${DEPENDENCY_DEFAULT_ARGS}")

  FOREACH(_dependency ${__dependency})
    SET(_name ${_dependency})

    # Check if dependency was not added already
    IF((NOT ${_nameUpper}_FOUND) AND (NOT ${_name}_FOUND) AND DEPENDENCY_FIND_PACKAGE)
      FIND_PACKAGE(${_name})
    ENDIF()

    string(TOUPPER ${_name} _nameUpper) # Uppercase is necessary due to FIND_PACKAGE_HANDLE_STANDARD_ARGS
    IF(${_nameUpper}_FOUND OR ${_name}_FOUND)
      # Append dependency and subdependencies to target, either exported or hidden
      IF(DEPENDENCY_EXPORT)
        list(APPEND ${icmaker_target}_MACRO_EXPORT_DEPENDENCIES ${_name})
      ELSE()
        list(APPEND ${icmaker_target}_MACRO_DEPENDENCIES ${_name})
      ENDIF()
      ICMAKER_RECURSIVE_DEPENDENCIES(${_name})
    ELSEIF(DEPENDENCY_OPTIONAL)
      LIST(APPEND ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING ${_name})
    ELSE()
      SET(${icmaker_target}_DEPENDENCIES_MATCHED FALSE)
      LIST(APPEND ${icmaker_target}_DEPENDENCIES_MISSING ${_name})
    ENDIF()
  ENDFOREACH(_dependency)
  
  IF(${icmaker_target}_MACRO_EXPORT_DEPENDENCIES)
    SET(${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES} CACHE INTERNAL "")
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       EXTERNAL SYSTEM DEPENDENCIES:
#
#   Fill one list: 
#     * ${icmaker_target}_MACRO_DEPENDENCIES: Dependencies used by target
#
#   Each dependency listed is simply added to the dependencies list without
#   any further checks.
# ----------------------------------------------------------------------------
MACRO(ICMAKER_SYSTEM_DEPENDENCIES)
  SET(${icmaker_target}_MACRO_TARGET_LINK_LIBRARIES ${${icmaker_target}_MACRO_TARGET_LINK_LIBRARIES} ${ARGN})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       BUILD A LIBRARY AND INSTALL IT INTO SUBDIR:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_LIBRARY_IN_SUBDIR _subdir_lib _subdir_bin _sources)
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    SET(${icmaker_target}_FOUND TRUE CACHE INTERNAL "")
    SET(${icmaker_target}_ICMAKER_PROJECT TRUE CACHE INTERNAL "")
    
    LIST(APPEND ${icmaker_target}_TARGET_DEPENDENCIES ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C} ${${icmaker_target}_MACRO_DEPENDENCIES})
    IF(${icmaker_target}_TARGET_DEPENDENCIES)
      LIST(REMOVE_DUPLICATES ${icmaker_target}_TARGET_DEPENDENCIES)
    ENDIF()
    
    IF(ICMAKER_VERBOSE)
      message(STATUS "  Dependencies for ${icmaker_target}:")
      message(STATUS "      (export) ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C}")
      message(STATUS "      (hidden) ${${icmaker_target}_MACRO_DEPENDENCIES}")
      message(STATUS "")
      message(STATUS "II. Preflight target ${icmaker_target}")
    ENDIF()
    ICMAKER_DEPENDECY_LIBS_AND_FLAGS(${${icmaker_target}_TARGET_DEPENDENCIES})

    # Setup target properties (definitions, includes, link libraries)
    LIST(APPEND ${icmaker_target}_AGGREGATE_DEFINITIONS ${${icmaker_target}_MACRO_DEFINITIONS} ${${icmaker_target}_MACRO_EXPORT_DEFINITIONS})
    LIST(APPEND ${icmaker_target}_AGGREGATE_INCLUDE_DIRS ${${icmaker_target}_MACRO_INCLUDE_DIRS} ${${icmaker_target}_MACRO_EXPORT_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    LIST(APPEND ${icmaker_target}_AGGREGATE_IDL_DIRS ${${icmaker_target}_MACRO_IDL_DIRS} ${${icmaker_target}_MACRO_EXPORT_IDL_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    LIST(APPEND ${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES ${${icmaker_target}_MACRO_TARGET_LINK_LIBRARIES})
    LIST(APPEND ${icmaker_target}_AGGREGATE_LDFLAGS ${${icmaker_target}_MACRO_LDFLAGS})

    LIST(APPEND ${icmaker_target}_AGGREGATE_SOURCES ${_sources})

    # Include directories need to be set before generating CUDA OBJs.
    include_directories(${${icmaker_target}_AGGREGATE_INCLUDE_DIRS})

    # See if there are CUDA object files to be added
    IF(CUDA_FOUND AND ${icmaker_target}_CUDA_FILES)
      SET(__cuda_generated_files)
      IF(${CMAKE_VERSION} VERSION_LESS "3.6.2") 
        CUDA_WRAP_SRCS(${icmaker_target}_CUDA_TARGET OBJ __cuda_generated_files ${${icmaker_target}_CUDA_FILES} OPTIONS ${ICMAKER_CUDA_CPPDEFINES})
      ELSE()
        CUDA_WRAP_SRCS(${icmaker_target}_CUDA_TARGET OBJ __cuda_generated_files ${${icmaker_target}_CUDA_FILES} OPTIONS ${ICMAKER_CUDA_CPPDEFINES} PHONY)
      ENDIF()
      LIST(APPEND ${icmaker_target}_AGGREGATE_SOURCES ${__cuda_generated_files})
    ENDIF()

    # Add sources to build target
    add_library(${icmaker_target} ${${icmaker_target}_AGGREGATE_SOURCES})

    # Set target dependencies
    IF(${icmaker_target}_TARGET_DEPENDENCIES)
      FOREACH(dependency ${${icmaker_target}_TARGET_DEPENDENCIES})
        IF(${dependency}_ICMAKER_PROJECT)
          add_dependencies(${icmaker_target} ${dependency})
        ENDIF()
      ENDFOREACH()
    ENDIF()

    # Add tcmalloc library if this is desired globally.
    SET(tcmalloc_libraries)
    SET(tcmalloc_flags)
    IF (ICMAKER_USE_TCMALLOC AND Tcmalloc_FOUND)
      SET(tcmalloc_libraries ${Tcmalloc_LIBRARY})
      SET(tcmalloc_flags ${Tcmalloc_FLAGS})
    ENDIF ()

    # set_target_properties below needs the definitions as a whitespace delimited string!
    SET(definitions_str ${tcmalloc_flags})
    FOREACH (definition ${${icmaker_target}_AGGREGATE_DEFINITIONS})
      SET(definitions_str "${definitions_str} ${definition}")
    ENDFOREACH()
    IF (definitions_str)
      set_target_properties(${icmaker_target} PROPERTIES COMPILE_FLAGS ${definitions_str})
    ENDIF()

    target_link_libraries(${icmaker_target} ${${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES} ${${icmaker_target}_AGGREGATE_LDFLAGS} ${tcmalloc_libraries})

    # Install definied target
    install(TARGETS ${icmaker_target}
            RUNTIME DESTINATION ${_subdir_bin} COMPONENT main
            LIBRARY DESTINATION ${_subdir_lib} COMPONENT main
            ARCHIVE DESTINATION ${_subdir_lib} COMPONENT main)

    SET_PROPERTY(
      TARGET ${icmaker_target}
      PROPERTY FOLDER "${${icmaker_target}_IDE_FOLDER_PREFIX}/${${icmaker_target}_IDE_FOLDER}")

    SET(${icmaker_package}_COMPONENTS ${${icmaker_package}_COMPONENTS} ${icmaker_target} CACHE INTERNAL "")

    # add export definitions to package definitions to be used in cmake package config
    SET(${icmaker_package}_DEFINITIONS ${${icmaker_package}_DEFINITIONS} ${${icmaker_target}_MACRO_EXPORT_DEFINITIONS} CACHE INTERNAL "")


    IF(ICMAKER_VERBOSE)
      message(STATUS "III. Building LIBRARY ${icmaker_target} into dir '${_subdir_lib}':")
      message(STATUS "    dependencies: ${${icmaker_target}_TARGET_DEPENDENCIES}")
      message(STATUS "    libraries:    ${${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES}")
      message(STATUS "    ldflags:      ${${icmaker_target}_AGGREGATE_LDFLAGS}")
      message(STATUS "    includes:     ${${icmaker_target}_AGGREGATE_INCLUDE_DIRS}")
      message(STATUS "    IDL includes: ${${icmaker_target}_AGGREGATE_IDL_DIRS}")
      message(STATUS "    definitions:  ${${icmaker_target}_AGGREGATE_DEFINITIONS}")
      message(STATUS "")
    ENDIF()
    IF(DEFINED ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING AND ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING)
      MESSAGE(STATUS "Info:    ${icmaker_target} -- building library without support for [${${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING}].")
    ENDIF()
  ELSE()
    MESSAGE(STATUS "Warning: ${icmaker_target} -- not building library, missing [${${icmaker_target}_DEPENDENCIES_MISSING}].")
    SET(${icmaker_target}_FOUND FALSE CACHE INTERNAL "")
  ENDIF()
ENDMACRO()


# ----------------------------------------------------------------------------
#                       BUILD A SET OF SWIG MODULES:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_SWIG_MODULES)
  PARSE_ARGUMENTS(SWIG_MODULES
    ""
    ""
    ${ARGN}
    )

  IF(ICMAKER_VERBOSE)
    message(STATUS "  Creating SWIG MODULES for target ${icmaker_target}")
  ENDIF()

  CAR(__swig_modules "${SWIG_MODULES_DEFAULT_ARGS}")

  IF(NOT DEFINED ${icmaker_target}_SWIG_FILE)
    MESSAGE(STATUS "Warning: ${icmaker_target} -- tried to run ICMAKER_BUILD_SWIG_MODULES but no input file was defined using ICMAKER_ADD_SWIG_FILE().")
  ELSE()
    IF(NOT SWIG_FOUND)
      MESSAGE(STATUS "Warning: ${icmaker_target} -- not building SWIG modules. Missing [SWIG].")
    ELSE()

      INCLUDE(${SWIG_USE_FILE})
      SET(CMAKE_SWIG_FLAGS "")

      SET_SOURCE_FILES_PROPERTIES(${${icmaker_target}_SWIG_FILE} PROPERTIES CPLUSPLUS ON)

      # we don't want to use this as long as it is possible.
#      SET_SOURCE_FILES_PROPERTIES(${${icmaker_target}_SWIG_FILE} PROPERTIES SWIG_FLAGS "-cpperraswarn")


      FOREACH(_swig_module ${__swig_modules})
        SET(_name ${_swig_module})

        string(TOLOWER ${_swig_module} _module)


        # set icmaker_target to the created swig module and save the parent one before.
        SET(icmaker_target_parent ${icmaker_target})
        SET(icmaker_target ${icmaker_target}_${_module})

        # add original target to the export dependencies
        ICMAKER_DEPENDENCIES(EXPORT ${icmaker_target_parent})


        # Add output-specific dependencies
        IF(${_module} STREQUAL "python")
          # add module-specific libs to the export dependencies>
          ICMAKER_DEPENDENCIES(EXPORT
              PythonLibs
          )
        ELSEIF(${_module} STREQUAL "java")
          # add module-specific libs to the export dependencies>
          ICMAKER_DEPENDENCIES(EXPORT
              JNI
          )
          MESSAGE(STATUS "Warning: ${icmaker_target_parent} -- tried to run ICMAKER_BUILD_SWIG_MODULES but this output is not completely supported yet: ${_swig_module}")
        ELSEIF(${_module} STREQUAL "perl")
          # add module-specific libs to the export dependencies>
          ICMAKER_DEPENDENCIES(EXPORT
              PerlLibs
          )
          MESSAGE(STATUS "Warning: ${icmaker_target_parent} -- tried to run ICMAKER_BUILD_SWIG_MODULES but this output is not completely supported yet: ${_swig_module}")
        ELSE()
          MESSAGE(STATUS "Warning: ${icmaker_target_parent} -- tried to run ICMAKER_BUILD_SWIG_MODULES but this output is not explicitly supported yet: ${_swig_module}")
        ENDIF()


        ###### DEPENDENCY HANDLING
        # This part is very similar to the one in ICMAKER_BUILD_LIBRARY_IN_SUBDIR
        # and ICMAKER_BUILD_PROGRAM_INTERNAL. Consider merging them.

        IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
          SET(${icmaker_target}_FOUND TRUE CACHE INTERNAL "")
          SET(${icmaker_target}_ICMAKER_PROJECT TRUE CACHE INTERNAL "")

          LIST(APPEND ${icmaker_target}_TARGET_DEPENDENCIES ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C} ${${icmaker_target}_MACRO_DEPENDENCIES})
          IF(${icmaker_target}_TARGET_DEPENDENCIES)
            LIST(REMOVE_DUPLICATES ${icmaker_target}_TARGET_DEPENDENCIES)
          ENDIF()

          IF(ICMAKER_VERBOSE)
            message(STATUS "  Dependencies for ${icmaker_target}:")
            message(STATUS "      (export) ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C}")
            message(STATUS "      (hidden) ${${icmaker_target}_MACRO_DEPENDENCIES}")
            message(STATUS "")
            message(STATUS "II. Preflight target ${icmaker_target}")
          ENDIF()
          ICMAKER_DEPENDECY_LIBS_AND_FLAGS(${${icmaker_target}_TARGET_DEPENDENCIES})


          # Setup target properties (definitions, includes, link libraries)
          LIST(APPEND ${icmaker_target}_AGGREGATE_DEFINITIONS ${${icmaker_target}_MACRO_DEFINITIONS} ${${icmaker_target}_MACRO_EXPORT_DEFINITIONS})
          LIST(APPEND ${icmaker_target}_AGGREGATE_INCLUDE_DIRS ${${icmaker_target}_MACRO_INCLUDE_DIRS} ${${icmaker_target}_MACRO_EXPORT_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
          LIST(APPEND ${icmaker_target}_AGGREGATE_IDL_DIRS ${${icmaker_target}_MACRO_IDL_DIRS} ${${icmaker_target}_MACRO_EXPORT_IDL_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
          LIST(APPEND ${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES ${${icmaker_target}_MACRO_TARGET_LINK_LIBRARIES})
          LIST(APPEND ${icmaker_target}_AGGREGATE_LDFLAGS ${${icmaker_target}_MACRO_LDFLAGS})


          ####### SETUP

          include_directories(${${icmaker_target}_AGGREGATE_INCLUDE_DIRS})

          IF(${CMAKE_VERSION} VERSION_LESS "3.8")
            SWIG_ADD_MODULE(${icmaker_target} ${_module} ${${icmaker_target_parent}_SWIG_FILE})
          ELSE()
            SWIG_ADD_LIBRARY(${icmaker_target} LANGUAGE ${_module} SOURCES ${${icmaker_target_parent}_SWIG_FILE})
          ENDIF()

          SWIG_LINK_LIBRARIES(${icmaker_target} ${${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES})


          ####### INSTALL

          IF(${_module} STREQUAL "python")
            SET_TARGET_PROPERTIES(_${icmaker_target} PROPERTIES OUTPUT_NAME _${icmaker_target_parent})

            # NOTE: this could be placed at a more central place
            # get the python module path where we want to install to
            execute_process ( COMMAND ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print( sysconfig.get_python_lib( plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}' ) )"
                              OUTPUT_VARIABLE _ABS_PYTHON_MODULE_PATH
                              OUTPUT_STRIP_TRAILING_WHITESPACE )

            # install lib and python file to python module path
            install(TARGETS _${icmaker_target} DESTINATION ${_ABS_PYTHON_MODULE_PATH})
            install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${icmaker_target_parent}.py DESTINATION ${_ABS_PYTHON_MODULE_PATH})
          ENDIF()


        ELSE()
          MESSAGE(STATUS "Warning: ${icmaker_target} -- not building library, missing [${${icmaker_target}_DEPENDENCIES_MISSING}].")
          SET(${icmaker_target}_FOUND FALSE CACHE INTERNAL "")
        ENDIF()


        # reset icmaker_target to the parent
        SET(icmaker_target ${icmaker_target_parent})
      ENDFOREACH(_swig_module)

    ENDIF()
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       BUILD A LIBRARY:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_LIBRARY)
  SET(${icmaker_target}_IDE_FOLDER_PREFIX "1. Libs")
  ICMAKER_BUILD_LIBRARY_IN_SUBDIR("lib" "bin" "${${icmaker_target}_SOURCES};${${icmaker_target}_HEADERS};${${icmaker_target}_GENERATED_SOURCES};${${icmaker_target}_GENERATED_HEADERS};${${icmaker_target}_GENERATED_CUDA_OBJS}")
  string(TOUPPER ${icmaker_target} icmaker_targetUpper)
  
  SET(${icmaker_targetUpper}_LIBRARIES "${icmaker_target}"  CACHE INTERNAL "")
  SET(${icmaker_targetUpper}_INCLUDE_DIRS "${${icmaker_target}_MACRO_EXPORT_INCLUDE_DIRS}" CACHE INTERNAL "")
  SET(${icmaker_targetUpper}_IDL_DIRS "${${icmaker_target}_MACRO_EXPORT_IDL_DIRS}" CACHE INTERNAL "")
  SET(${icmaker_targetUpper}_DEFINITIONS "${${icmaker_target}_MACRO_EXPORT_DEFINITIONS}" CACHE INTERNAL "")

  SOURCE_GROUP("Sources" FILES ${${icmaker_target}_SOURCES})
  SOURCE_GROUP("Includes" FILES ${${icmaker_target}_HEADERS})
  SOURCE_GROUP("Generated Sources" FILES ${${icmaker_target}_GENERATED_SOURCES})
  SOURCE_GROUP("Generated Includes" FILES ${${icmaker_target}_GENERATED_HEADERS})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       BUILD A PLUGIN:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_PLUGIN _subdir)
  SET(${icmaker_target}_IDE_FOLDER_PREFIX "3. Plugins")
  ICMAKER_BUILD_LIBRARY_IN_SUBDIR("plugins/${_subdir}" "plugins/${_subdir}" "${${icmaker_target}_SOURCES};${${icmaker_target}_HEADERS};${${icmaker_target}_GENERATED_SOURCES};${${icmaker_target}_GENERATED_HEADERS};${${icmaker_target}_GENERATED_CUDA_OBJS}")
  SOURCE_GROUP("Sources" FILES ${${icmaker_target}_SOURCES})
  SOURCE_GROUP("Includes" FILES ${${icmaker_target}_HEADERS})
  SOURCE_GROUP("Generated Sources" FILES ${${icmaker_target}_GENERATED_SOURCES})
  SOURCE_GROUP("Generated Includes" FILES ${${icmaker_target}_GENERATED_HEADERS})
ENDMACRO()

# ----------------------------------------------------------------------------
#                       BUILD AN ANNOUNCEMENT:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_ANNOUNCEMENT)
  SET(${icmaker_target}_IDE_FOLDER_PREFIX "1. Libs")
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    SET(${icmaker_target}_FOUND TRUE CACHE INTERNAL "")
    string(TOUPPER ${icmaker_target} icmaker_targetUpper)
    add_custom_target(${icmaker_target} SOURCES ${${icmaker_target}_HEADERS})
    
    LIST(APPEND ${icmaker_target}_TARGET_DEPENDENCIES ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C} ${${icmaker_target}_MACRO_DEPENDENCIES})
    IF(${icmaker_target}_TARGET_DEPENDENCIES)
      LIST(REMOVE_DUPLICATES ${icmaker_target}_TARGET_DEPENDENCIES)
    ENDIF()
    IF(ICMAKER_VERBOSE)
      message(STATUS "  Dependencies for ${icmaker_target}:")
      message(STATUS "      (export) ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C}")
      message(STATUS "      (hidden) ${${icmaker_target}_MACRO_DEPENDENCIES}")
      message(STATUS "")
      message(STATUS "II. Preflight target ${icmaker_target}")
    ENDIF()
    ICMAKER_DEPENDECY_LIBS_AND_FLAGS(${${icmaker_target}_TARGET_DEPENDENCIES})
    
    SET(${icmaker_targetUpper}_LIBRARIES ""  CACHE INTERNAL "")
    SET(${icmaker_targetUpper}_INCLUDE_DIRS "${${icmaker_target}_MACRO_EXPORT_INCLUDE_DIRS}" CACHE INTERNAL "")
    SET(${icmaker_targetUpper}_DEFINITIONS "${${icmaker_target}_MACRO_EXPORT_DEFINITIONS}" CACHE INTERNAL "")

    # add export definitions to package definitions to be used in cmake package config
    SET(${icmaker_package}_DEFINITIONS ${${icmaker_package}_DEFINITIONS} ${${icmaker_target}_MACRO_EXPORT_DEFINITIONS} CACHE INTERNAL "")

  ELSE()
    MESSAGE(STATUS "Warning: ${icmaker_target} -- not building announcement, missing [${${icmaker_target}_DEPENDENCIES_MISSING}].")
    SET(${icmaker_target}_FOUND FALSE CACHE INTERNAL "")
  ENDIF()

  IF (DEFINED ${icmaker_target})
    SET_PROPERTY(
      TARGET ${icmaker_target}
      PROPERTY FOLDER "5. Announcements/${${icmaker_target}_IDE_FOLDER}")
  ENDIF()

  IF(ICMAKER_VERBOSE)
    message(STATUS "III. Building ANNOUNCEMENT ${_target}:")
    message(STATUS "    dependencies: ${${icmaker_target}_TARGET_DEPENDENCIES}")
    message(STATUS "    libraries:    ${${icmaker_targetUpper}_LIBRARIES}")
    message(STATUS "    includes:     ${${icmaker_targetUpper}_INCLUDE_DIRS}")
    message(STATUS "    definitions:  ${${icmaker_targetUpper}_DEFINITIONS}")
    message(STATUS "")
  ENDIF()
  SOURCE_GROUP("Includes" FILES ${${icmaker_target}_HEADERS})
ENDMACRO()


# ----------------------------------------------------------------------------
#              INTERNAL HELPER MACRO TO BUILD A PROGRAM OR TEST:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_PROGRAM_INTERNAL)
    SET(${icmaker_target}_ICMAKER_PROJECT TRUE CACHE INTERNAL "")
    
    LIST(APPEND ${icmaker_target}_TARGET_DEPENDENCIES ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C} ${${icmaker_target}_MACRO_DEPENDENCIES})
    IF(${icmaker_target}_TARGET_DEPENDENCIES)
      LIST(REMOVE_DUPLICATES ${icmaker_target}_TARGET_DEPENDENCIES)
    ENDIF()
    IF(ICMAKER_VERBOSE)
      message(STATUS "  Dependencies for ${icmaker_target}:")
      message(STATUS "      (export) ${${icmaker_target}_MACRO_EXPORT_DEPENDENCIES_C}")
      message(STATUS "      (hidden) ${${icmaker_target}_MACRO_DEPENDENCIES}")
      message(STATUS "")
      message(STATUS "II. Preflight target ${icmaker_target}")
    ENDIF()
    ICMAKER_DEPENDECY_LIBS_AND_FLAGS(${${icmaker_target}_TARGET_DEPENDENCIES})

    # Setup target properties (definitions, includes, link libraries)
    LIST(APPEND ${icmaker_target}_AGGREGATE_DEFINITIONS ${${icmaker_target}_MACRO_DEFINITIONS} ${${icmaker_target}_MACRO_EXPORT_DEFINITIONS})
    LIST(APPEND ${icmaker_target}_AGGREGATE_INCLUDE_DIRS ${${icmaker_target}_MACRO_INCLUDE_DIRS} ${${icmaker_target}_MACRO_EXPORT_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    LIST(APPEND ${icmaker_target}_AGGREGATE_IDL_DIRS ${${icmaker_target}_MACRO_IDL_DIRS} ${${icmaker_target}_MACRO_EXPORT_IDL_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    LIST(APPEND ${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES ${${icmaker_target}_MACRO_TARGET_LINK_LIBRARIES})
    LIST(APPEND ${icmaker_target}_AGGREGATE_LDFLAGS ${${icmaker_target}_MACRO_LDFLAGS})

    # Include directories need to be set before generating CUDA OBJs.
    include_directories(${${icmaker_target}_AGGREGATE_INCLUDE_DIRS})

    # See if there are CUDA object files to be added
    IF(CUDA_FOUND AND ${icmaker_target}_CUDA_FILES)
      SET(__cuda_generated_files)
      IF(${CMAKE_VERSION} VERSION_LESS "3.6.2") 
        CUDA_WRAP_SRCS(${icmaker_target}_CUDA_TARGET OBJ __cuda_generated_files ${${icmaker_target}_CUDA_FILES} OPTIONS ${ICMAKER_CUDA_CPPDEFINES})
      ELSE()
        CUDA_WRAP_SRCS(${icmaker_target}_CUDA_TARGET OBJ __cuda_generated_files ${${icmaker_target}_CUDA_FILES} OPTIONS ${ICMAKER_CUDA_CPPDEFINES} PHONY)
      ENDIF()
      LIST(APPEND ${icmaker_target}_GENERATED_CUDA_OBJS ${__cuda_generated_files})
    ENDIF()

    # Add sources to build target
    add_executable(${icmaker_target} ${${icmaker_target}_SOURCES} ${${icmaker_target}_HEADERS} ${${icmaker_target}_GENERATED_SOURCES} ${${icmaker_target}_GENERATED_HEADERS} ${${icmaker_target}_GENERATED_CUDA_OBJS})
    
    # Set target dependencies
    foreach(dependency ${${icmaker_target}_TARGET_DEPENDENCIES})
        IF(${dependency}_ICMAKER_PROJECT)
            add_dependencies(${icmaker_target} ${dependency})
        ENDIF()
    endforeach()

    # Add tcmalloc library if this is desired globally.
    SET(tcmalloc_libraries)
    SET(tcmalloc_flags)
    IF (ICMAKER_USE_TCMALLOC AND Tcmalloc_FOUND)
      SET(tcmalloc_libraries ${Tcmalloc_LIBRARY})
      SET(tcmalloc_flags ${Tcmalloc_FLAGS})
    ENDIF ()

    # set_target_properties below needs the definitions as a whitespace delimited string!
    SET(definitions_str ${tcmalloc_flags})
    FOREACH (definition ${${icmaker_target}_AGGREGATE_DEFINITIONS})
      SET(definitions_str "${definitions_str} ${definition}")
    ENDFOREACH()
    IF (definitions_str)
      set_target_properties(${icmaker_target} PROPERTIES COMPILE_FLAGS ${definitions_str})
    ENDIF()

    target_link_libraries(${icmaker_target} ${${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES} ${${icmaker_target}_AGGREGATE_LDFLAGS} ${tcmalloc_libraries})

    # Install definied target
    install(TARGETS ${icmaker_target}
            RUNTIME DESTINATION bin COMPONENT main
            LIBRARY DESTINATION lib COMPONENT main
            ARCHIVE DESTINATION lib COMPONENT main)

    SET_PROPERTY(TARGET ${icmaker_target}
                 PROPERTY FOLDER "${${icmaker_target}_IDE_FOLDER_PREFIX}/${${icmaker_target}_IDE_FOLDER}")

    SOURCE_GROUP("Sources" FILES ${${icmaker_target}_SOURCES})
    SOURCE_GROUP("Includes" FILES ${${icmaker_target}_HEADERS})
    SOURCE_GROUP("Generated Sources" FILES ${${icmaker_target}_GENERATED_SOURCES})
    SOURCE_GROUP("Generated Includes" FILES ${${icmaker_target}_GENERATED_HEADERS})
ENDMACRO()


# ----------------------------------------------------------------------------
#                       BUILD A PROGRAM:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_PROGRAM)
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    SET(${icmaker_target}_IDE_FOLDER_PREFIX "2. Programs")
    ICMAKER_BUILD_PROGRAM_INTERNAL()

    IF(ICMAKER_VERBOSE)
      message(STATUS "III. Building PROGRAM ${icmaker_target}:")
      message(STATUS "    libraries:    ${${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES}")
      message(STATUS "    ldflags:      ${${icmaker_target}_AGGREGATE_LDFLAGS}")
      message(STATUS "    includes:     ${${icmaker_target}_AGGREGATE_INCLUDE_DIRS}")
      message(STATUS "    IDL includes: ${${icmaker_target}_AGGREGATE_IDL_DIRS}")
      message(STATUS "    definitions:  ${${icmaker_target}_AGGREGATE_DEFINITIONS}")
      message(STATUS "")
    ENDIF()
    IF(DEFINED ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING AND ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING)
      MESSAGE(STATUS "Info:    ${icmaker_target} -- building executable without support for [${${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING}].")
    ENDIF()
  ELSE()
    MESSAGE(STATUS "Warning: ${icmaker_target} -- not building executable, missing [${${icmaker_target}_DEPENDENCIES_MISSING}].")
  ENDIF()
ENDMACRO()


# ----------------------------------------------------------------------------
#                       BUILD A CUSTOM TARGET:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_CUSTOM _group _sources)
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    ADD_CUSTOM_TARGET(${icmaker_target} SOURCES ${_sources})
    SET_PROPERTY(TARGET ${icmaker_target}
                 PROPERTY FOLDER "6. Custom Targets/${${icmaker_target}_IDE_FOLDER}")
    SOURCE_GROUP(${_group} FILES ${_sources})

  ELSE()
    MESSAGE(STATUS "Warning: ${icmaker_target} -- not building custom target, missing [${${icmaker_target}_DEPENDENCIES_MISSING}].")
  ENDIF()
ENDMACRO()


# ----------------------------------------------------------------------------
#                       BUILD A TEST:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_BUILD_TEST)
  IF(BUILD_TESTS)
    IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
      SET(${icmaker_target}_IDE_FOLDER_PREFIX "4. Unit Tests")
      ICMAKER_BUILD_PROGRAM_INTERNAL()

      ADD_TEST(NAME ${icmaker_target}
               COMMAND ${EXECUTABLE_OUTPUT_PATH}/${icmaker_target})

      IF(ICMAKER_VERBOSE)
        message(STATUS "III. Building PROGRAM ${icmaker_target}:")
        message(STATUS "    libraries:    ${${icmaker_target}_AGGREGATE_TARGET_LINK_LIBRARIES}")
        message(STATUS "    ldflags:      ${${icmaker_target}_AGGREGATE_LDFLAGS}")
        message(STATUS "    includes:     ${${icmaker_target}_AGGREGATE_INCLUDE_DIRS}")
        message(STATUS "    IDL includes: ${${icmaker_target}_AGGREGATE_IDL_DIRS}")
        message(STATUS "    definitions:  ${${icmaker_target}_AGGREGATE_DEFINITIONS}")
        message(STATUS "")
      ENDIF()
    IF(DEFINED ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING AND ${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING)
      MESSAGE(STATUS "Info:    ${icmaker_target} -- building test without support for [${${icmaker_target}_OPTIONAL_DEPENDENCIES_MISSING}].")
    ENDIF()
    ELSE()
      MESSAGE(STATUS "Warning: ${icmaker_target} -- not building test, missing [${${icmaker_target}_DEPENDENCIES_MISSING}].")
    ENDIF()
  ENDIF()
ENDMACRO()


# ----------------------------------------------------------------------------
#                       INSTALL HEADERS:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_INSTALL_HEADERS _subdir)
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    INSTALL(FILES ${${icmaker_target}_HEADERS} DESTINATION include/${_subdir} COMPONENT main)
  ENDIF()
ENDMACRO()

MACRO(ICMAKER_INSTALL_GLOBHEADERS _subdir)
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION include/${_subdir} COMPONENT main FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       INSTALL HEADER EXTRAS:
# ----------------------------------------------------------------------------
# Usage: ICMAKER_INSTALL_HEADER_EXTRAS(subdir file1 file2 ...)
# Installs the given files into include/subdir.
MACRO(ICMAKER_INSTALL_HEADER_EXTRAS _subdir)
  PARSE_ARGUMENTS(ADD_EXTRAS "" "" ${ARGN})
  CAR(__extras "${ADD_EXTRAS_DEFAULT_ARGS}")
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    INSTALL(FILES ${__extras} DESTINATION include/${_subdir} COMPONENT main)
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       INSTALL SHARED EXTRAS:
# ----------------------------------------------------------------------------
# Usage: ICMAKER_INSTALL_SHARED_EXTRAS(subdir file1 file2 ...)
# Installs the given files into share/subdir.
MACRO(ICMAKER_INSTALL_SHARED_EXTRAS _subdir)
  PARSE_ARGUMENTS(ADD_EXTRAS "" "" ${ARGN})
  CAR(__extras "${ADD_EXTRAS_DEFAULT_ARGS}")
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    INSTALL(FILES ${__extras} DESTINATION share/${_subdir} COMPONENT main)
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       INSTALL CONFIG EXTRAS:
# ----------------------------------------------------------------------------
# Usage: ICMAKER_INSTALL_CONFIG_EXTRAS(subdir file1 file2 ...)
# Installs the given files into etc/subdir.
MACRO(ICMAKER_INSTALL_CONFIG_EXTRAS _subdir)
  PARSE_ARGUMENTS(ADD_EXTRAS "" "" ${ARGN})
  CAR(__extras "${ADD_EXTRAS_DEFAULT_ARGS}")
  IF(NOT DEFINED ${icmaker_target}_DEPENDENCIES_MATCHED OR ${icmaker_target}_DEPENDENCIES_MATCHED)
    INSTALL(FILES ${__extras} DESTINATION etc/${_subdir} COMPONENT main)
  ENDIF()
ENDMACRO()

# ----------------------------------------------------------------------------
#                       GENERATE VERSION.H:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_CONFIGURE_VERSION_H)
  FILE(READ "${CMAKE_CURRENT_SOURCE_DIR}/Version.txt" VERSION_STR)
  STRING(REGEX REPLACE "(.*)\\.(.*)\\.(.*)" "\\1" VERSION_MAJOR ${VERSION_STR})
  STRING(REGEX REPLACE "(.*)\\.(.*)\\.(.*)" "\\2" VERSION_MINOR ${VERSION_STR})
  STRING(REGEX REPLACE "(.*)\\.(.*)\\.(.*)" "\\3" VERSION_BUILD ${VERSION_STR})
  CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/Version.h.in" "${CMAKE_CURRENT_BINARY_DIR}/Version.h")
ENDMACRO()

# ----------------------------------------------------------------------------
#                       OpenMP Flags:
# ----------------------------------------------------------------------------
MACRO(SET_OPENMP_FLAGS)
  IF(OPENMP_FOUND)
    # Only bitch when OpenMP is actually there
    MESSAGE(AUTHOR_WARNING "Calling SET_OPENMP_FLAGS() is not needed anymore, please remove it from ${CMAKE_CURRENT_LIST_FILE}")
  ENDIF()
ENDMACRO()


# ----------------------------------------------------------------------------
#                       CMake Config File:
# ----------------------------------------------------------------------------
# Simplified from OpenCV's packaging code.
# Don't forget to provide the config file ${icmaker_package}-config.cmake.in
MACRO(ICMAKER_CONFIGURE_PACKAGE)
  set(ICLIB_PACKAGE_NAME_CONFIGCMAKE "${icmaker_package}")
  set(ICLIB_PACKAGE_NAME_CONFIGCMAKE_COMPONENTS "${${icmaker_package}_COMPONENTS}")
  set(ICLIB_PACKAGE_NAME_CONFIGCMAKE_DEFINITIONS "${${icmaker_package}_DEFINITIONS}")

  #  Part 1/3: ${CURRENT_BIN_DIR}/${icmaker_package}-config.cmake              -> For use *without* "make install"
  set(ICLIB_INCLUDE_DIRS_CONFIGCMAKE "\"${CMAKE_CURRENT_SOURCE_DIR}/src\"")
  set(ICLIB_LIB_DIRS_CONFIGCMAKE "${LIBRARY_OUTPUT_PATH}")

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${icmaker_package}-config.cmake.in")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${icmaker_package}-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${icmaker_package}-config.cmake" IMMEDIATE @ONLY)
  else()
    configure_file("${ICMAKER_DIRECTORY}/icmaker_template-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${icmaker_package}-config.cmake" IMMEDIATE @ONLY)
  endif()

  # We do not want to make the packages registered in the User Package Registry to prevent unwanted
  # finding/linking, especially when using multiple workspaces. That's why we deactivate it here. If
  # this implies other flaws please consider a new solution!
  #
  # More:
  # https://cmake.org/cmake/help/v3.0/command/export.html
  # https://ids-git.fzi.de/core/robot_folders/issues/30
  #
  #EXPORT(PACKAGE ${icmaker_package})

  #  Part 2/3: ${BIN_DIR}/unix-install/${icmaker_package}-config.cmake -> For use *with* "make install"
  set(ICLIB_INCLUDE_DIRS_CONFIGCMAKE "\"\${${icmaker_package}_INSTALL_PATH}/include\"")
  set(ICLIB_LIB_DIRS_CONFIGCMAKE "\"\${${icmaker_package}_INSTALL_PATH}/lib\"")

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${icmaker_package}-config.cmake.in")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${icmaker_package}-config.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/${icmaker_package}-config.cmake" IMMEDIATE @ONLY)
  else()
    configure_file("${ICMAKER_DIRECTORY}/icmaker_template-config.cmake.in" "${CMAKE_BINARY_DIR}/unix-install/${icmaker_package}-config.cmake" IMMEDIATE @ONLY)
  endif()

  if(UNIX)
    #http://www.vtk.org/Wiki/CMake/Tutorials/Packaging reference
    # For a command "find_package(<name> [major[.minor]] [EXACT] [REQUIRED|QUIET])"
    # cmake will look in the following dir on unix:
    #                <prefix>/(share|lib)/cmake/<name>*/                     (U)
    #                <prefix>/(share|lib)/<name>*/                           (U)
    #                <prefix>/(share|lib)/<name>*/(cmake|CMake)/             (U)
    install(FILES "${CMAKE_BINARY_DIR}/unix-install/${icmaker_package}-config.cmake" DESTINATION share/${icmaker_package}/)
  endif()
ENDMACRO()
