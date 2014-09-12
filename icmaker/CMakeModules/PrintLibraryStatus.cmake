# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
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
