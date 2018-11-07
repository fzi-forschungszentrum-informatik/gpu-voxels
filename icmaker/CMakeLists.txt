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

cmake_minimum_required(VERSION 2.8.3)

# This can be removed as soon as old scripts don't expect this file to be the
# entrance point any more.
if(DEFINED ICMAKER_DIRECTORY)
  if(${ICMAKER_DIRECTORY} STREQUAL ${CMAKE_CURRENT_LIST_DIR})
  message( FATAL_ERROR "
The main CMakeLists.txt inside the main icmaker folder is not meant to be
used for including IcMaker any more. Use this instead:
   include(UseFullIcMaker)
... or make sure you have the most recent version of your repositories.
  ")
  endif()
endif()


project(icmaker)

# ----------------------------------------------------------------------------
# SPECIAL VERSIONS OF ICMAKER MACROS - maybe these could be merged with icmaker
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
#                       DEFINE THE PACKAGE:
# ----------------------------------------------------------------------------
MACRO(ICMAKER_REGISTER_PACKAGE _package)
  SET(icmaker_package "${_package}")
  SET(${icmaker_package}_COMPONENTS "" CACHE INTERNAL "")
  SET(${icmaker_package}_DEFINITIONS "" CACHE INTERNAL "")
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

  ############## THIS LINE IS SPECIAL ##################
  set(ICLIB_CMAKE_MODULE_DIRS_CONFIGCMAKE "\"${CMAKE_CURRENT_SOURCE_DIR}\"")
  ######################################################

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${icmaker_package}-config.cmake.in")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${icmaker_package}-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${icmaker_package}-config.cmake" IMMEDIATE @ONLY)
  else()
    configure_file("${ICMAKER_DIRECTORY}/icmaker_template-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/${icmaker_package}-config.cmake" IMMEDIATE @ONLY)
  endif()
  EXPORT(PACKAGE ${icmaker_package})

  #  Part 2/3: ${BIN_DIR}/unix-install/${icmaker_package}-config.cmake -> For use *with* "make install"
  set(ICLIB_INCLUDE_DIRS_CONFIGCMAKE "\"\${${icmaker_package}_INSTALL_PATH}/include\"")
  set(ICLIB_LIB_DIRS_CONFIGCMAKE "\"\${${icmaker_package}_INSTALL_PATH}/lib\"")

  ############## THIS LINE IS SPECIAL ##################
  set(ICLIB_CMAKE_MODULE_DIRS_CONFIGCMAKE "\"\${${icmaker_package}_INSTALL_PATH}/share/${icmaker_package}\"")
  ######################################################

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



# ----------------------------------------------------------------------------
# Actual Script
# ----------------------------------------------------------------------------

# Register for packaging.
ICMAKER_REGISTER_PACKAGE(icmaker)

# install icmaker cmake files
install(FILES "CorePackages.cmake" "IcMaker.cmake" "IcMakerMacros.cmake" "icmaker_template-config.cmake.in"
        DESTINATION share/${icmaker_package})
install(DIRECTORY "CMakeModules"
        DESTINATION share/${icmaker_package})

# Configure packaging. Don't forget to provide the config file <package_name>-config.cmake.in
ICMAKER_CONFIGURE_PACKAGE()
