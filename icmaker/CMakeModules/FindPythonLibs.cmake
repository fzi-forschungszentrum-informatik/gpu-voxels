# - Find python libraries
# This module finds if Python is installed and determines where the
# include files and libraries are. It also determines what the name of
# the library is. This code sets the following variables:
#
#  PythonLIBS_FOUND           - have the Python libs been found
#  Python_LIBRARIES           - path to the python library
#  Python_INCLUDE_PATH        - path to where Python.h is found (deprecated)
#  Python_INCLUDE_DIRS        - path to where Python.h is found
#  Python_DEBUG_LIBRARIES     - path to the debug library
#  Python_ADDITIONAL_VERSIONS - list of additional Python versions to search for

# Based on CMake's own FindPythonLibs module.  Adapted to icmaker by
# Jan Oberlaender, 2011-11-18.
#=============================================================================
# Copyright (c) 2002 Kitware, Inc., Insight Consortium
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * The names of Kitware, Inc., the Insight Consortium, or the names of
#    any consortium members, or of any contributors, may not be used to
#    endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

INCLUDE(CMakeFindFrameworks)
# Search for the python framework on Apple.
CMAKE_FIND_FRAMEWORKS(Python)

# Set up the versions we know about, in the order we will search. Always add
# the user supplied additional versions to the front.
set(_Python_VERSIONS
  ${Python_ADDITIONAL_VERSIONS}
  2.7 2.6 2.5 2.4 2.3 2.2 2.1 2.0 1.6 1.5)

FOREACH(_CURRENT_VERSION ${_Python_VERSIONS})
  STRING(REPLACE "." "" _CURRENT_VERSION_NO_DOTS ${_CURRENT_VERSION})
  IF(WIN32)
    FIND_LIBRARY(Python_DEBUG_LIBRARY
      NAMES python${_CURRENT_VERSION_NO_DOTS}_d python
      PATHS
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs/Debug
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs )
  ENDIF(WIN32)

  FIND_LIBRARY(Python_LIBRARY
    NAMES python${_CURRENT_VERSION_NO_DOTS} python${_CURRENT_VERSION}
    PATHS
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/libs
    # Avoid finding the .dll in the PATH.  We want the .lib.
    NO_SYSTEM_ENVIRONMENT_PATH
  )
  # Look for the static library in the Python config directory
  FIND_LIBRARY(Python_LIBRARY
    NAMES python${_CURRENT_VERSION_NO_DOTS} python${_CURRENT_VERSION}
    # Avoid finding the .dll in the PATH.  We want the .lib.
    NO_SYSTEM_ENVIRONMENT_PATH
    # This is where the static library is usually located
    PATH_SUFFIXES python${_CURRENT_VERSION}/config
  )

  # For backward compatibility, honour value of Python_INCLUDE_PATH, if
  # Python_INCLUDE_DIR is not set.
  IF(DEFINED Python_INCLUDE_PATH AND NOT DEFINED Python_INCLUDE_DIR)
    SET(Python_INCLUDE_DIR "${Python_INCLUDE_PATH}" CACHE PATH
      "Path to where Python.h is found" FORCE)
  ENDIF(DEFINED Python_INCLUDE_PATH AND NOT DEFINED Python_INCLUDE_DIR)

  SET(Python_FRAMEWORK_INCLUDES)
  IF(Python_FRAMEWORKS AND NOT Python_INCLUDE_DIR)
    FOREACH(dir ${Python_FRAMEWORKS})
      SET(Python_FRAMEWORK_INCLUDES ${Python_FRAMEWORK_INCLUDES}
        ${dir}/Versions/${_CURRENT_VERSION}/include/python${_CURRENT_VERSION})
    ENDFOREACH(dir)
  ENDIF(Python_FRAMEWORKS AND NOT Python_INCLUDE_DIR)

  FIND_PATH(Python_INCLUDE_DIR
    NAMES Python.h
    PATHS
      ${Python_FRAMEWORK_INCLUDES}
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Python\\PythonCore\\${_CURRENT_VERSION}\\InstallPath]/include
    PATH_SUFFIXES
      python${_CURRENT_VERSION}
  )

  # For backward compatibility, set Python_INCLUDE_PATH, but make it internal.
  SET(Python_INCLUDE_PATH "${Python_INCLUDE_DIR}" CACHE INTERNAL
    "Path to where Python.h is found (deprecated)")

ENDFOREACH(_CURRENT_VERSION)

MARK_AS_ADVANCED(
  Python_DEBUG_LIBRARY
  Python_LIBRARY
  Python_INCLUDE_DIR
)

# We use Python_INCLUDE_DIR, Python_LIBRARY and Python_DEBUG_LIBRARY for the
# cache entries because they are meant to specify the location of a single
# library. We now set the variables listed by the documentation for this
# module.
SET(Python_INCLUDE_DIRS "${Python_INCLUDE_DIR}")
SET(Python_LIBRARIES "${Python_LIBRARY}")
SET(Python_DEBUG_LIBRARIES "${Python_DEBUG_LIBRARY}")

# PythonLibs_INCLUDE_DIRS, PythonLibs_LIBRARIES and
# PythonLibs_DEBUG_LIBRARIES are necessary so that we actually get the
# desired results for the compiler command line.
SET(PythonLibs_INCLUDE_DIRS "${Python_INCLUDE_DIR}")
SET(PythonLibs_LIBRARIES "${Python_LIBRARY}")
SET(PythonLibs_DEBUG_LIBRARIES "${Python_DEBUG_LIBRARY}")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PythonLibs DEFAULT_MSG Python_LIBRARIES Python_INCLUDE_DIRS)


# Python_ADD_MODULE(<name> src1 src2 ... srcN) is used to build modules for python.
# Python_WRITE_MODULES_HEADER(<filename>) writes a header file you can include
# in your sources to initialize the static python modules
FUNCTION(Python_ADD_MODULE _NAME )
  GET_PROPERTY(_TARGET_SUPPORTS_SHARED_LIBS
    GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
  OPTION(Python_ENABLE_MODULE_${_NAME} "Add module ${_NAME}" TRUE)
  OPTION(Python_MODULE_${_NAME}_BUILD_SHARED
    "Add module ${_NAME} shared" ${_TARGET_SUPPORTS_SHARED_LIBS})

  # Mark these options as advanced
  MARK_AS_ADVANCED(Python_ENABLE_MODULE_${_NAME}
    Python_MODULE_${_NAME}_BUILD_SHARED)

  IF(Python_ENABLE_MODULE_${_NAME})
    IF(Python_MODULE_${_NAME}_BUILD_SHARED)
      SET(PY_MODULE_TYPE MODULE)
    ELSE(Python_MODULE_${_NAME}_BUILD_SHARED)
      SET(PY_MODULE_TYPE STATIC)
      SET_PROPERTY(GLOBAL  APPEND  PROPERTY  PY_STATIC_MODULES_LIST ${_NAME})
    ENDIF(Python_MODULE_${_NAME}_BUILD_SHARED)

    SET_PROPERTY(GLOBAL  APPEND  PROPERTY  PY_MODULES_LIST ${_NAME})
    ADD_LIBRARY(${_NAME} ${PY_MODULE_TYPE} ${ARGN})
#    TARGET_LINK_LIBRARIES(${_NAME} ${Python_LIBRARIES})

    IF(Python_MODULE_${_NAME}_BUILD_SHARED)
      SET_TARGET_PROPERTIES(${_NAME} PROPERTIES PREFIX "${Python_MODULE_PREFIX}")
      IF(WIN32 AND NOT CYGWIN)
        SET_TARGET_PROPERTIES(${_NAME} PROPERTIES SUFFIX ".pyd")
      ENDIF(WIN32 AND NOT CYGWIN)
    ENDIF(Python_MODULE_${_NAME}_BUILD_SHARED)

  ENDIF(Python_ENABLE_MODULE_${_NAME})
ENDFUNCTION(Python_ADD_MODULE)

FUNCTION(Python_WRITE_MODULES_HEADER _filename)

  GET_PROPERTY(PY_STATIC_MODULES_LIST  GLOBAL  PROPERTY PY_STATIC_MODULES_LIST)

  GET_FILENAME_COMPONENT(_name "${_filename}" NAME)
  STRING(REPLACE "." "_" _name "${_name}")
  STRING(TOUPPER ${_name} _nameUpper)
  SET(_filename ${CMAKE_CURRENT_BINARY_DIR}/${_filename})

  SET(_filenameTmp "${_filename}.in")
  FILE(WRITE ${_filenameTmp} "/*Created by cmake, do not edit, changes will be lost*/\n")
  FILE(APPEND ${_filenameTmp}
"#ifndef ${_nameUpper}
#define ${_nameUpper}

#include <Python.h>

#ifdef __cplusplus
extern \"C\" {
#endif /* __cplusplus */

")

  FOREACH(_currentModule ${PY_STATIC_MODULES_LIST})
    FILE(APPEND ${_filenameTmp} "extern void init${Python_MODULE_PREFIX}${_currentModule}(void);\n\n")
  ENDFOREACH(_currentModule ${PY_STATIC_MODULES_LIST})

  FILE(APPEND ${_filenameTmp}
"#ifdef __cplusplus
}
#endif /* __cplusplus */

")


  FOREACH(_currentModule ${PY_STATIC_MODULES_LIST})
    FILE(APPEND ${_filenameTmp} "int ${_name}_${_currentModule}(void) \n{\n  static char name[]=\"${Python_MODULE_PREFIX}${_currentModule}\"; return PyImport_AppendInittab(name, init${Python_MODULE_PREFIX}${_currentModule});\n}\n\n")
  ENDFOREACH(_currentModule ${PY_STATIC_MODULES_LIST})

  FILE(APPEND ${_filenameTmp} "void ${_name}_LoadAllPythonModules(void)\n{\n")
  FOREACH(_currentModule ${PY_STATIC_MODULES_LIST})
    FILE(APPEND ${_filenameTmp} "  ${_name}_${_currentModule}();\n")
  ENDFOREACH(_currentModule ${PY_STATIC_MODULES_LIST})
  FILE(APPEND ${_filenameTmp} "}\n\n")
  FILE(APPEND ${_filenameTmp} "#ifndef EXCLUDE_LOAD_ALL_FUNCTION\nvoid CMakeLoadAllPythonModules(void)\n{\n  ${_name}_LoadAllPythonModules();\n}\n#endif\n\n#endif\n")

# with CONFIGURE_FILE() cmake complains that you may not use a file created using FILE(WRITE) as input file for CONFIGURE_FILE()
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_filenameTmp}" "${_filename}" OUTPUT_QUIET ERROR_QUIET)

ENDFUNCTION(Python_WRITE_MODULES_HEADER)
