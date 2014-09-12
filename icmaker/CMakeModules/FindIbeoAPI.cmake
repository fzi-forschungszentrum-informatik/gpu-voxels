# - Try to find IbeoAPI
# Once done this will define
#
#  IbeoAPI_FOUND - System has IbeoAPI
#  IbeoAPI_INCLUDE_DIRS - IbeoAPI include directories (multiple directories!)
#  IbeoAPI_LIBRARIES - Link these to use IbeoAPI
#  IbeoAPI_S_LIBRARIES - Link these to use IbeoAPI
#  IbeoAPI_LIBRARY_DIRS - The path to where the IbeoAPI library files are.
#
#  IbeoAPI_LIBRARY             The IbeoAPI base libary.
#  IbeoAPI_ARCNET_LIBRARY             The IbeoAPI ARCNET libary.
#  IbeoAPI_CAN_LIBRARY             The IbeoAPI CAN Time libary.
#  IbeoAPI_INTERN_LIBRARY             The IbeoAPI intern (device) libary.
#
#  IbeoAPI_S_LIBRARY             The static IbeoAPI base libary.
#  IbeoAPI_ARCNET_S_LIBRARY             The static IbeoAPI ARCNET libary.
#  IbeoAPI_CAN_S_LIBRARY             The static IbeoAPI CAN libary.
#  IbeoAPI_INTERN_S_LIBRARY             The static IbeoAPI intern libary.
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#
# Options that can be defined beforehand or chosen by the user:
#
#  IbeoAPI_SHARED
#  IbeoAPI_STATIC
#  IbeoAPI_WITH_CAN
# (see below for their documentation)
#


option (IbeoAPI_SHARED "Use IbeoAPI as a shared library." OFF)
option (IbeoAPI_STATIC "Use IbeoAPI as a static library." ON)

if (NOT IbeoAPI_SHARED AND NOT IbeoAPI_STATIC)
  message (SEND_ERROR "Oops: Neither static nor shared build of IbeoAPI was enabled. Enable at least one of them.")
endif (NOT IbeoAPI_SHARED AND NOT IbeoAPI_STATIC)

if (UNIX)
  option (IbeoAPI_WITH_CAN "Use IbeoAPI with CAN interface." ON)
else (UNIX)
  option (IbeoAPI_WITH_CAN "Use IbeoAPI with CAN interface." OFF)
endif (UNIX)


if (IbeoAPI_LIBRARIES AND IbeoAPI_INCLUDE_DIRS)
  # in cache already
  set(IbeoAPI_FOUND TRUE)
else (IbeoAPI_LIBRARIES AND IbeoAPI_INCLUDE_DIRS)

  INCLUDE (MacroAppendForeach)

  # ########################################
  if (WIN32)
    set(IbeoAPI_INCLUDE_SEARCH_DIRS
      $ENV{IBEOAPIINCLUDEDIR}
      C:/ibeoapi/include
      "C:/Programme/ibeoapi/include"
      "C:/Program Files/ibeoapi/include"
      /usr/include
    )

    set(IbeoAPI_LIBRARIES_SEARCH_DIRS
      C:/ibeoapi/lib
      "C:/Programme/ibeoapi/lib"
      "C:/Program Files/ibeoapi/lib"
      /usr/lib
    )

    if (MSVC71 OR MSVC80 OR MSVC90)
		# There is no CMAKE_BUILD_TYPE in MSVC
        set(IbeoAPI_LIBRARIES_SUFFIXES
		  -2
		  -d-2
          -1
          -d-1
        )
    endif (MSVC71 OR MSVC80 OR MSVC90)

    if (MINGW)
      if (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(IbeoAPI_LIBRARIES_SUFFIXES
		  -d-2
		  -2
          -d-1
          -1
        )
      else (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(IbeoAPI_LIBRARIES_SUFFIXES
		  -2
          -1
        )
      endif (CMAKE_BUILD_TYPE STREQUAL Debug)
    endif (MINGW)

    if (CYGWIN)
      if (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(IbeoAPI_LIBRARIES_SUFFIXES
          -gcc-mt-d
        )
      else (CMAKE_BUILD_TYPE STREQUAL Debug)
        set(IbeoAPI_LIBRARIES_SUFFIXES
          -gcc-mt
        )
      endif (CMAKE_BUILD_TYPE STREQUAL Debug)
    endif (CYGWIN)

  # ########################################
  else (WIN32)

    set(IbeoAPI_INCLUDE_SEARCH_DIRS
      $ENV{IBEOAPIINCLUDEDIR}
      ${CMAKE_SOURCE_DIR}/../../Modules
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include
    )

    set(IbeoAPI_LIBRARIES_SUFFIXES
    )
  endif (WIN32)
  # End of OS-specific search choices
  # ########################################

  # Find the path to the headers
  find_path(IbeoAPI_INCLUDE_DIR
    NAMES
      IbeoAPI/Types.hpp
    PATHS
      ${IbeoAPI_INCLUDE_SEARCH_DIRS}
      ${CMAKE_SOURCE_DIR}/../../Modules/IbeoAPI
      ${CMAKE_SOURCE_DIR}/../../../Modules/IbeoAPI
      ${IbeoAPI_DIR}/..
    PATH_SUFFIXES
      ${IbeoAPI_PATH_SUFFIX}
	DOC "Path where the IbeoAPI/Types.hpp header file is located, i.e. one directory above the actual files"
  )

  # Determine build directory name to use this as yet another search
  # location for the libraries. We use the build directory name either
  # from the env variable IbeoAPI_BUILD_DIRNAME
  set (BUILD_DIRNAME $ENV{IbeoAPI_BUILD_DIRNAME})
  if (NOT BUILD_DIRNAME)
    # Or, if the env variable was empty, try to guess it from current
    # build directory's name
    file (RELATIVE_PATH BUILD_DIRNAME ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
  endif (NOT BUILD_DIRNAME)

  # Check whether we find the build directory of the IbeoAPI, which is
  # simply the directory containing CMakeCache.txt
  find_path (IbeoAPI_BUILD_DIR
	NAMES CMakeCache.txt
	PATHS
      $ENV{IBEOAPILIBDIR}
      ${IbeoAPI_DIR}
      ${IbeoAPI_DIR}/${BUILD_DIRNAME}
      ${IbeoAPI_DIR}/${CMAKE_BUILD_TYPE}
      ${IbeoAPI_DIR}/build
	DOC "Path to the build directory of the IbeoAPI used in this project"
  )

  # ########################################
  # Did we find the original IbeoAPI build directory? Then exploit the
  # cache variables from there as much as possible.

  if (IbeoAPI_BUILD_DIR)
	# Hah! We found the CMakeCache of the original IbeoAPI. We now
	# read this and try to use most of those variables directly!
	load_cache (${IbeoAPI_BUILD_DIR} READ_WITH_PREFIX IbeoAPI_
	  ENABLE_SHARED_BUILD
	  ENABLE_STATIC_BUILD
	  ENABLE_CAN_INTERFACE
	  )
	set (IbeoAPI_SHARED ${IbeoAPI_ENABLE_SHARED_BUILD})
	set (IbeoAPI_STATIC ${IbeoAPI_ENABLE_STATIC_BUILD})
	set (IbeoAPI_WITH_CAN ${IbeoAPI_ENABLE_CAN_INTERFACE})
	mark_as_advanced (
	  IbeoAPI_SHARED
	  IbeoAPI_STATIC
	  IbeoAPI_WITH_CAN
	  )
  endif (IbeoAPI_BUILD_DIR)

  # ########################################

  # We have found the header path, so add more search paths for
  # the library using this header path
  set(IbeoAPI_LIBRARIES_SEARCH_DIRS
    $ENV{IBEOAPILIBDIR}
    ${IbeoAPI_INCLUDE_DIR}/IbeoAPI
    ${IbeoAPI_INCLUDE_DIR}/../lib
    ${IbeoAPI_INCLUDE_DIR}/../lib/link-static
    ${IbeoAPI_INCLUDE_DIR}/IbeoAPI/${BUILD_DIRNAME}
    ${IbeoAPI_INCLUDE_DIR}/IbeoAPI/${CMAKE_BUILD_TYPE}
    ${IbeoAPI_INCLUDE_DIR}/IbeoAPI/build
    ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
    ${IbeoAPI_BUILD_DIR}
  )

  # The possible path suffixes
  set (LIBRARY_PATH_SUFFIXES ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})

  # ########################################

  # We need the subdirectories as search directories as well
  MACRO_APPEND_FOREACH (_SEARCHDIRS "" "/ibeointern" ${IbeoAPI_LIBRARIES_SEARCH_DIRS})
  MACRO_APPEND_FOREACH (_SEARCHDIRS "" "/ibeoarcnet" ${IbeoAPI_LIBRARIES_SEARCH_DIRS})
  MACRO_APPEND_FOREACH (_SEARCHDIRS "" "/ibeocan" ${IbeoAPI_LIBRARIES_SEARCH_DIRS})
  SET (IbeoAPI_LIBRARIES_SEARCH_DIRS ${IbeoAPI_LIBRARIES_SEARCH_DIRS} ${_SEARCHDIRS})

  # Now search for the libraries. The library names might have one of
  # the suffixes of the IbeoAPI_LIBRARIES_SUFFIXES list, which is
  # OS-specific.
  foreach (TMP_IbeoAPI_LIBRARIES_SUFFIX "" ${IbeoAPI_LIBRARIES_SUFFIXES})

    # #################
    if (IbeoAPI_SHARED)
      if (NOT IbeoAPI_LIBRARY)
        find_library (IbeoAPI_LIBRARY
          NAMES
            ibeoapi${TMP_IbeoAPI_LIBRARIES_SUFFIX}
          PATHS
            ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
          PATH_SUFFIXES
            ${LIBRARY_PATH_SUFFIXES}
        )
  
        if (IbeoAPI_LIBRARY)
          # IbeoAPI_LIBRARY was found
          # sets the libraries suffix, this code is ugly
          # but CMake does not allow to break a loop :/
          set(IbeoAPI_LIBRARIES_SUFFIX
            ${TMP_IbeoAPI_LIBRARIES_SUFFIX}
            CACHE INTERNAL "" FORCE
          )
        endif (IbeoAPI_LIBRARY)
  
      else (NOT IbeoAPI_LIBRARY)
        # This is needed if the variable was given on the cmake command
        # line so that it is cached for subsequent cmake runs.
        set (IbeoAPI_LIBRARY ${IbeoAPI_LIBRARY} CACHE FILEPATH "Path to a library.")
      endif (NOT IbeoAPI_LIBRARY)

      if (NOT IbeoAPI_ARCNET_LIBRARY)
        find_library(IbeoAPI_ARCNET_LIBRARY
          NAMES
            ibeoarcnet${TMP_IbeoAPI_LIBRARIES_SUFFIX}
          PATHS
            ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
          PATH_SUFFIXES
            ${LIBRARY_PATH_SUFFIXES}
        )
      else (NOT IbeoAPI_ARCNET_LIBRARY)
        set (IbeoAPI_ARCNET_LIBRARY ${IbeoAPI_ARCNET_LIBRARY} CACHE FILEPATH "Path to a library.")
      endif (NOT IbeoAPI_ARCNET_LIBRARY)
  
      if (NOT IbeoAPI_INTERN_LIBRARY)
        find_library(IbeoAPI_INTERN_LIBRARY
          NAMES
            ibeointern${TMP_IbeoAPI_LIBRARIES_SUFFIX}
          PATHS
            ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
          PATH_SUFFIXES
            ${LIBRARY_PATH_SUFFIXES}
        )
      else (NOT IbeoAPI_INTERN_LIBRARY)
        set (IbeoAPI_INTERN_LIBRARY ${IbeoAPI_INTERN_LIBRARY} CACHE FILEPATH "Path to a library.")
      endif (NOT IbeoAPI_INTERN_LIBRARY)
  
      if (IbeoAPI_WITH_CAN)
        if (NOT IbeoAPI_CAN_LIBRARY)
          find_library(IbeoAPI_CAN_LIBRARY
            NAMES
              ibeocan${TMP_IbeoAPI_LIBRARIES_SUFFIX}
            PATHS
              ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
            PATH_SUFFIXES
              ${LIBRARY_PATH_SUFFIXES}
          )
        else (NOT IbeoAPI_CAN_LIBRARY)
          set (IbeoAPI_CAN_LIBRARY ${IbeoAPI_CAN_LIBRARY} CACHE FILEPATH "Path to a library.")
        endif (NOT IbeoAPI_CAN_LIBRARY)
      endif (IbeoAPI_WITH_CAN)
  
    endif (IbeoAPI_SHARED)

    # #################
  
    if (IbeoAPI_STATIC)
  
      if (NOT IbeoAPI_S_LIBRARY)
        find_library(IbeoAPI_S_LIBRARY
          NAMES
            ibeoapi-static${TMP_IbeoAPI_LIBRARIES_SUFFIX}
          PATHS
            ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
          PATH_SUFFIXES
            ${LIBRARY_PATH_SUFFIXES}
        )
      else (NOT IbeoAPI_S_LIBRARY)
        set (IbeoAPI_S_LIBRARY ${IbeoAPI_S_LIBRARY} CACHE FILEPATH "Path to a library.")
      endif (NOT IbeoAPI_S_LIBRARY)
  
      if (NOT IbeoAPI_ARCNET_S_LIBRARY)
        find_library(IbeoAPI_ARCNET_S_LIBRARY
          NAMES
            ibeoarcnet-static${TMP_IbeoAPI_LIBRARIES_SUFFIX}
          PATHS
            ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
          PATH_SUFFIXES
            ${LIBRARY_PATH_SUFFIXES}
        )
      else (NOT IbeoAPI_ARCNET_S_LIBRARY)
        set (IbeoAPI_ARCNET_S_LIBRARY ${IbeoAPI_ARCNET_S_LIBRARY} CACHE FILEPATH "Path to a library.")
      endif (NOT IbeoAPI_ARCNET_S_LIBRARY)
  
      if (NOT IbeoAPI_INTERN_S_LIBRARY)
        find_library(IbeoAPI_INTERN_S_LIBRARY
          NAMES
            ibeointern-static${TMP_IbeoAPI_LIBRARIES_SUFFIX}
          PATHS
            ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
          PATH_SUFFIXES
            ${LIBRARY_PATH_SUFFIXES}
        )
      else (NOT IbeoAPI_INTERN_S_LIBRARY)
        set (IbeoAPI_INTERN_S_LIBRARY ${IbeoAPI_INTERN_S_LIBRARY} CACHE FILEPATH "Path to a library.")
      endif (NOT IbeoAPI_INTERN_S_LIBRARY)
  
      if (IbeoAPI_WITH_CAN)
        if (NOT IbeoAPI_CAN_S_LIBRARY)
          find_library(IbeoAPI_CAN_S_LIBRARY
            NAMES
              ibeocan-static${TMP_IbeoAPI_LIBRARIES_SUFFIX}
            PATHS
              ${IbeoAPI_LIBRARIES_SEARCH_DIRS}
            PATH_SUFFIXES
              ${LIBRARY_PATH_SUFFIXES}
          )
        else (NOT IbeoAPI_CAN_S_LIBRARY)
          set (IbeoAPI_CAN_S_LIBRARY ${IbeoAPI_CAN_S_LIBRARY} CACHE FILEPATH "Path to a library.")
        endif (NOT IbeoAPI_CAN_S_LIBRARY)
      endif (IbeoAPI_WITH_CAN)
  
    endif (IbeoAPI_STATIC)
    # #################

  endforeach (TMP_IbeoAPI_LIBRARIES_SUFFIX)
  # ########################################

  # Set the output variables
  set(IbeoAPI_INCLUDE_DIRS
    ${IbeoAPI_INCLUDE_DIR}
    ${IbeoAPI_INCLUDE_DIR}/IbeoAPI
  )


  if (IbeoAPI_LIBRARY OR IbeoAPI_S_LIBRARY)
    set(IbeoAPI_LIBRARIES
      ${IbeoAPI_LIBRARIES}
      ${IbeoAPI_LIBRARY}
      ${IbeoAPI_S_LIBRARY}
    )
  endif (IbeoAPI_LIBRARY OR IbeoAPI_S_LIBRARY)
  if (IbeoAPI_ARCNET_LIBRARY OR IbeoAPI_ARCNET_S_LIBRARY)
    set(IbeoAPI_LIBRARIES
      ${IbeoAPI_LIBRARIES}
      ${IbeoAPI_ARCNET_LIBRARY}
      ${IbeoAPI_ARCNET_S_LIBRARY}
    )
  endif (IbeoAPI_ARCNET_LIBRARY OR IbeoAPI_ARCNET_S_LIBRARY)
  if ((IbeoAPI_CAN_LIBRARY OR IbeoAPI_CAN_S_LIBRARY) AND IbeoAPI_WITH_CAN)
    set(IbeoAPI_LIBRARIES
      ${IbeoAPI_LIBRARIES}
      ${IbeoAPI_CAN_LIBRARY}
      ${IbeoAPI_CAN_S_LIBRARY}
    )
  endif ((IbeoAPI_CAN_LIBRARY OR IbeoAPI_CAN_S_LIBRARY) AND IbeoAPI_WITH_CAN)
  if (IbeoAPI_INTERN_LIBRARY OR IbeoAPI_INTERN_S_LIBRARY)
    set(IbeoAPI_LIBRARIES
      ${IbeoAPI_LIBRARIES}
      ${IbeoAPI_INTERN_LIBRARY}
      ${IbeoAPI_INTERN_S_LIBRARY}
    )
  endif (IbeoAPI_INTERN_LIBRARY OR IbeoAPI_INTERN_S_LIBRARY)

  if (IbeoAPI_INCLUDE_DIRS AND IbeoAPI_LIBRARIES)
    set(IbeoAPI_FOUND TRUE)
  endif (IbeoAPI_INCLUDE_DIRS AND IbeoAPI_LIBRARIES)

  foreach (IbeoAPI_LIBDIR ${IbeoAPI_LIBRARIES})
    get_filename_component(IbeoAPI_LIBRARY_DIRS ${IbeoAPI_LIBDIR} PATH)
  endforeach (IbeoAPI_LIBDIR ${IbeoAPI_LIBRARIES})

  if (IbeoAPI_FOUND)
    if (NOT IbeoAPI_FIND_QUIETLY)
      message(STATUS "Found IbeoAPI headers in: ${IbeoAPI_INCLUDE_DIRS}")
    endif (NOT IbeoAPI_FIND_QUIETLY)
  else (IbeoAPI_FOUND)
    if (IbeoAPI_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find IbeoAPI, but it is required.")
    endif (IbeoAPI_FIND_REQUIRED)
  endif (IbeoAPI_FOUND)

  # show the IbeoAPI_INCLUDE_DIRS and IbeoAPI_LIBRARIES variables only in the advanced view
  mark_as_advanced(IbeoAPI_INCLUDE_DIRS IbeoAPI_LIBRARIES IbeoAPI_LIBRARY_DIRS IbeoAPI_DEFINITIONS IbeoAPI_LIBRARIES_SUFFIX)

endif (IbeoAPI_LIBRARIES AND IbeoAPI_INCLUDE_DIRS)
