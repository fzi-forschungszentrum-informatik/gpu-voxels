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

# PRINT_LIBRARY_STATUS(<name> ...)
#
# This functions intends to print a message displaying the status of a library check.
# It uses the FindPackageMessage function, provided by cmake's default modules
#
# The function arguments should provide the name of the library as well as additional flags
#   PRINT_LIBRARY_STATUS(NAME
#     VERSION <version_var>
#     DETAILS "[<library_dirs>][<include_dirs>]"
#     COMPONENTS <library_components> )

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

MACRO(PRINT_LIBRARY_STATUS)

  INCLUDE(FindPackageMessage)
  PARSE_ARGUMENTS(MSG "VERSION;DETAILS;COMPONENTS" "" ${ARGN})
  CAR(_PREFIX ${MSG_DEFAULT_ARGS})

  IF(${_PREFIX}_FOUND)
    IF(MSG_VERSION)
      FIND_PACKAGE_MESSAGE(${_PREFIX}
                   "Found ${_PREFIX}: ... yes (Version: ${MSG_VERSION})" "${MSG_DETAILS}")
    ELSE()
      FIND_PACKAGE_MESSAGE(${_PREFIX}
                   "Found ${_PREFIX}: ... yes" "${MSG_DETAILS}")
    ENDIF()
  ELSE()
    IF(MSG_COMPONENTS)
      MESSAGE(STATUS "Could NOT find all components of ${_PREFIX} - missing (one or more components): ${MSG_COMPONENTS}")
    ELSE()
      MESSAGE(STATUS "Could NOT find ${_PREFIX}")
    ENDIF()
  ENDIF()
ENDMACRO()
