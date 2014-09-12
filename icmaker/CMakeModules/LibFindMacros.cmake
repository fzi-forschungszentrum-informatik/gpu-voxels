# Works the same as find_package, but forwards the "REQUIRED" and "QUIET" arguments
# used for the current package. For this to work, the first parameter must be the
# prefix of the current package, then the prefix of the new package etc, which are
# passed to find_package.
macro (libfind_package PREFIX)
  set (LIBFIND_PACKAGE_ARGS ${ARGN})
  if (${PREFIX}_FIND_QUIETLY)
    set (LIBFIND_PACKAGE_ARGS ${LIBFIND_PACKAGE_ARGS} QUIET)
  endif (${PREFIX}_FIND_QUIETLY)
  if (${PREFIX}_FIND_REQUIRED)
    set (LIBFIND_PACKAGE_ARGS ${LIBFIND_PACKAGE_ARGS} REQUIRED)
  endif (${PREFIX}_FIND_REQUIRED)
  find_package(${LIBFIND_PACKAGE_ARGS})
endmacro (libfind_package)

# CMake developers made the UsePkgConfig system deprecated in the same release (2.6)
# where they added pkg_check_modules. Consequently I need to support both in my scripts
# to avoid those deprecated warnings. Here's a helper that does just that.
# Works identically to pkg_check_modules, except that no checks are needed prior to use.
macro (libfind_pkg_check_modules PREFIX PKGNAME)
  if (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} EQUAL 4)
    include(UsePkgConfig)
    pkgconfig(${PKGNAME} ${PREFIX}_INCLUDE_DIRS ${PREFIX}_LIBRARY_DIRS ${PREFIX}_LDFLAGS ${PREFIX}_CFLAGS)
  else (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} EQUAL 4)
    find_package(PkgConfig)
    if (PKG_CONFIG_FOUND)
      pkg_check_modules(${PREFIX} QUIET ${PKGNAME})
    endif (PKG_CONFIG_FOUND)
  endif (${CMAKE_MAJOR_VERSION} EQUAL 2 AND ${CMAKE_MINOR_VERSION} EQUAL 4)
endmacro (libfind_pkg_check_modules)

# Do the final processing once the paths have been detected.
# If include dirs are needed, ${PREFIX}_PROCESS_INCLUDES should be set to contain
# all the variables, each of which contain one include directory.
# Ditto for ${PREFIX}_PROCESS_LIBS and library files.
# Will set ${PREFIX}_FOUND, ${PREFIX}_INCLUDE_DIRS and ${PREFIX}_LIBRARIES.
# Also handles errors in case library detection was required, etc.
macro (libfind_process PREFIX)
  # Skip processing if already processed during this run
  if (NOT ${PREFIX}_FOUND)
    # Start with the assumption that the library was found
    set (${PREFIX}_FOUND TRUE)

    # Process all includes and set _FOUND to false if any are missing
    foreach (i ${${PREFIX}_PROCESS_INCLUDES})
      if (${i})
        set (${PREFIX}_INCLUDE_DIRS ${${PREFIX}_INCLUDE_DIRS} ${${i}})
        mark_as_advanced(${i})
      else (${i})
        set (${PREFIX}_FOUND FALSE)
      endif (${i})
    endforeach (i)

    # Process all libraries and set _FOUND to false if any are missing
    foreach (i ${${PREFIX}_PROCESS_LIBS})
      if (${i})
        set (${PREFIX}_LIBRARIES ${${PREFIX}_LIBRARIES} ${${i}})
        mark_as_advanced(${i})
      else (${i})
        set (${PREFIX}_FOUND FALSE)
      endif (${i})
    endforeach (i)

    # Print message and/or exit on fatal error
    if (${PREFIX}_FOUND)
      if (NOT ${PREFIX}_FIND_QUIETLY)
        # message (STATUS "Found ${PREFIX} ${${PREFIX}_VERSION}")
      endif (NOT ${PREFIX}_FIND_QUIETLY)
    else (${PREFIX}_FOUND)
      if (${PREFIX}_FIND_REQUIRED)
        foreach (i ${${PREFIX}_PROCESS_INCLUDES} ${${PREFIX}_PROCESS_LIBS})
          message("${i}=${${i}}")
        endforeach (i)
        message (FATAL_ERROR "Required library ${PREFIX} NOT FOUND.\nInstall the library (dev version) and try again. If the library is already installed, use ccmake to set the missing variables manually.")
      endif (${PREFIX}_FIND_REQUIRED)
    endif (${PREFIX}_FOUND)
  endif (NOT ${PREFIX}_FOUND)
endmacro (libfind_process)

macro(libfind_library PREFIX basename)
  set(TMP "")
  if(MSVC80)
    set(TMP -vc80)
  endif(MSVC80)
  if(MSVC90)
    set(TMP -vc90)
  endif(MSVC90)
  set(${PREFIX}_LIBNAMES ${basename}${TMP})
  if(${ARGC} GREATER 2)
    set(${PREFIX}_LIBNAMES ${basename}${TMP}-${ARGV2})
    string(REGEX REPLACE "\\." "_" TMP ${${PREFIX}_LIBNAMES})
    set(${PREFIX}_LIBNAMES ${${PREFIX}_LIBNAMES} ${TMP})
  endif(${ARGC} GREATER 2)
  find_library(${PREFIX}_LIBRARY
    NAMES ${${PREFIX}_LIBNAMES}
    PATHS ${${PREFIX}_PKGCONF_LIBRARY_DIRS}
  )
endmacro(libfind_library)

# A more universal library check which combines various methods of
# discovering and checking.  It looks for the given HEADERS, LIBRARIES
# and EXECUTABLES at various locations:
#
# - If <name>_ROOT is set by the user (e.g. with cmake
#   -DFOO_ROOT=/opt/foo), that prefix is searched first.
# - Otherwise, if pkg-config reports something, the paths reported by
#   it are searched first.
# - Other paths, as given by the programmer using the HINTS parameter,
#   are searched next.  E.g., if "HINTS /foo/bar" is provided, the
#   check will search for headers in /foo/bar/include, libraries in
#   /foo/bar/lib, and executables in /foo/bar/bin.
# - Further paths given by HEADER_PATHS, LIBRARY_PATHS and
#   EXECUTABLE_PATHS are searched next for headers, libraries or
#   executables, respectively.
# - The default paths /usr and /usr/local are searched last.
#
# All required headers must be found in the same location.  E.g., if
# one required header is found under /usr/local/include/foo, but
# another one only exists at a different location such as
# /opt/foo/include/foo, the library is not found because these headers
# would likely be inconsistent.  The folder in which the very first
# header is found has to contain the remaining headers as well.  The
# same goes for libraries and executables.
#
# If a library is found successfully, a precompiler definition
# "-D_IC_BUILDER_<name>_" is set.
#
# Usage: libfind_lib_with_pkg_config(
#          name pkgconfig-name
#          HEADERS hdr1 hdr2 ...
#          LIBRARIES lib1 lib2 ...
#          EXECUTABLES exec1 exec2 ...
#          HINTS hint1 hint2 ...
#          HEADER_PATHS path1 path2 ...
#          LIBRARY_PATHS path1 path2 ...
#          EXECUTABLE_PATHS path1 path2 ...
#          DEFINE definition)
#
# Parameters:
#  HEADERS          ... Required headers.
#  LIBRARIES        ... Required libraries.
#  EXECUTABLES      ... Required executables.
#  HINTS            ... Path prefixes to search under.
#  HEADER_PATHS     ... Paths where to search headers.
#  LIBRARY_PATHS    ... Paths where to search libraries.
#  EXECUTABLE_PATHS ... Paths where to search executables.
#  DEFINE           ... Name of the preprocessor macro to define if
#                       the library was found (default:
#                       _IC_BUILDER_<name>_).
#
# Provided variables:
#  <name>_FOUND: Set if the library was found.  If so, the following
#                additional variables are set.
#  <name>_INCLUDE_DIRS   ... the '-I' preprocessor flags (w/o the '-I')
#  <name>_LIBRARY_DIRS   ... the paths of the libraries (w/o the '-L')
# Variables provided if pkg-config was employed:
#  <name>_DEFINITIONS    ... Preprocessor definitions.
#  <name>_LIBRARIES      ... only the libraries (w/o the '-l')
#  <name>_LDFLAGS        ... all required linker flags
#  <name>_LDFLAGS_OTHER  ... all other linker flags
#  <name>_CFLAGS         ... all required cflags
#  <name>_CFLAGS_OTHER   ... the other compiler flags
#  <name>_VERSION        ... version of the module
#  <name>_PREFIX         ... prefix-directory of the module
#  <name>_INCLUDEDIR     ... include-dir of the module
#  <name>_LIBDIR         ... lib-dir of the module

macro(libfind_lib_with_pkg_config)
  # Get all arguments
  parse_arguments(LIBFIND
    "HEADERS;LIBRARIES;EXECUTABLES;HINTS;HEADER_PATHS;LIBRARY_PATHS;EXECUTABLE_PATHS;DEFINE"
    ""
    ${ARGN})
  car(NAME ${LIBFIND_DEFAULT_ARGS})
  cdr(LIBFIND_DEFAULT_ARGS ${LIBFIND_DEFAULT_ARGS})
  car(PKGCONF_NAME ${LIBFIND_DEFAULT_ARGS})
  list(APPEND LIBFIND_HINTS /usr /usr/local)
  list(REMOVE_DUPLICATES LIBFIND_HINTS)
  foreach (p ${LIBFIND_HINTS})
    list(APPEND LIBFIND_HEADER_PATHS ${p}/include)
    list(APPEND LIBFIND_LIBRARY_PATHS ${p}/lib)
    list(APPEND LIBFIND_EXECUTABLE_PATHS ${p}/bin)
  endforeach ()
  if (NOT LIBFIND_DEFINE)
    set(LIBFIND_DEFINE "_IC_BUILDER_${NAME}_")
  endif ()

  if (${NAME}_FOUND)
    # In cache already
    set(${NAME}_FIND_QUIETLY TRUE)
  endif ()

  # If a root is set, use that, otherwise use pkg-config output
  if ((NOT DEFINED ${NAME}_ROOT OR ${NAME}_ROOT STREQUAL "")
      AND ("$ENV{${NAME}_ROOT}" STREQUAL ""))
    libfind_pkg_check_modules(${NAME}_PKGCONF ${PKGCONF_NAME})
    if (${NAME}_PKGCONF_FOUND)
      if (${NAME}_PKGCONF_INCLUDE_DIRS)
        list(INSERT LIBFIND_HEADER_PATHS 0 ${${NAME}_PKGCONF_INCLUDE_DIRS})
      endif ()
      if (${NAME}_PKGCONF_LIBRARY_DIRS)
        list(INSERT LIBFIND_LIBRARY_PATHS 0 ${${NAME}_PKGCONF_LIBRARY_DIRS})
      endif ()
      list(INSERT LIBFIND_EXECUTABLE_PATHS 0 "${${NAME}_PKGCONF_PREFIX}/bin")
      if (${NAME}_PKGCONF_LIBRARIES)
        set(LIBFIND_LIBRARIES ${${NAME}_PKGCONF_LIBRARIES})
      endif ()
    endif ()
  else ()
    list(INSERT LIBFIND_HEADER_PATHS 0 "${${NAME}_ROOT}/include")
    list(INSERT LIBFIND_LIBRARY_PATHS 0 "${${NAME}_ROOT}/lib")
    list(INSERT LIBFIND_EXECUTABLE_PATHS 0 "${${NAME}_ROOT}/bin")
  endif ()
  if (LIBFIND_HEADER_PATHS)
    list(REMOVE_DUPLICATES LIBFIND_HEADER_PATHS)
  endif ()
  if (LIBFIND_LIBRARY_PATHS)
    list(REMOVE_DUPLICATES LIBFIND_LIBRARY_PATHS)
  endif ()
  if (LIBFIND_EXECUTABLE_PATHS)
    list(REMOVE_DUPLICATES LIBFIND_EXECUTABLE_PATHS)
  endif ()

  # Now check for headers.  Note that we expect each header to reside
  # in the same location.
  foreach (h ${LIBFIND_HEADERS})
    unset(LIBFIND_INCLUDE_DIR)
    unset(LIBFIND_INCLUDE_DIR CACHE)
    find_path(LIBFIND_INCLUDE_DIR
      NAMES ${h}
      PATHS ${LIBFIND_HEADER_PATHS})
    if (LIBFIND_INCLUDE_DIR)
      set(LIBFIND_HEADER_PATHS ${LIBFIND_INCLUDE_DIR})
    else ()
      message(STATUS "${NAME}: Could not find required header file ${h}.")
      return ()
    endif ()
  endforeach ()
  set(LIBFIND_FOUND_INCLUDE_DIR ${LIBFIND_INCLUDE_DIR})

  # Now check for libraries.  Note that we expect each library to
  # reside in the same location.
  unset(LIBFIND_FOUND_LIBRARIES)
  unset(LIBFIND_FOUND_LIBRARIES CACHE)
  set(LIBFIND_FOUND_LIBRARIES "")
  foreach (l ${LIBFIND_LIBRARIES})
    unset(LIBFIND_LIBRARY_PATH)
    unset(LIBFIND_LIBRARY_PATH CACHE)
    find_library(LIBFIND_LIBRARY_PATH
      NAMES ${l}
      PATHS ${LIBFIND_LIBRARY_PATHS})
    if (LIBFIND_LIBRARY_PATH)
      list(APPEND LIBFIND_FOUND_LIBRARIES "${l}")
      get_filename_component(LIBFIND_LIBRARY_DIR ${LIBFIND_LIBRARY_PATH} PATH)
      set(LIBFIND_LIBRARY_PATHS ${LIBFIND_LIBRARY_DIR})
    else ()
      message(STATUS "${NAME}: Could not find required library ${l}.")
      return ()
    endif ()
  endforeach ()

  # Now check for executables.  Note that we expect each executable to
  # reside in the same location.
  foreach (e ${LIBFIND_EXECUTABLES})
    unset(LIBFIND_EXECUTABLE_PATH)
    unset(LIBFIND_EXECUTABLE_PATH CACHE)
    find_program(LIBFIND_EXECUTABLE_PATH
      NAMES ${e}
      PATHS ${LIBFIND_EXECUTABLE_PATHS})
    if (LIBFIND_EXECUTABLE_PATH)
      get_filename_component(LIBFIND_EXECUTABLE_DIR ${LIBFIND_EXECUTABLE_PATH} PATH)
      set(LIBFIND_EXECUTABLE_PATHS ${LIBFIND_EXECUTABLE_DIR})
    else ()
      message(STATUS "${NAME}: Could not find required executable ${e}.")
      return ()
    endif ()
  endforeach ()

  # Everything seems to be OK.  Set the relevant variables.
  if (LIBFIND_FOUND_INCLUDE_DIR)
    set(${NAME}_PROCESS_INCLUDES LIBFIND_FOUND_INCLUDE_DIR)
  endif ()
  if (LIBFIND_FOUND_LIBRARIES)
    set(${NAME}_PROCESS_LIBS LIBFIND_FOUND_LIBRARIES)
  endif ()
  libfind_process(${NAME})

  if (${NAME}_FOUND)
    set(${NAME}_LIBRARY_DIRS ${${NAME}_PKGCONF_LIBRARY_DIRS} CACHE INTERNAL "")
    set(${NAME}_LDFLAGS ${${NAME}_PKGCONF_LDFLAGS} CACHE INTERNAL "")
    set(${NAME}_LDFLAGS_OTHER ${${NAME}_PKGCONF_LDFLAGS_OTHER} CACHE INTERNAL "")
    set(${NAME}_CFLAGS ${${NAME}_PKGCONF_CFLAGS} CACHE INTERNAL "")
    set(${NAME}_CFLAGS_OTHER ${${NAME}_PKGCONF_CFLAGS_OTHER} CACHE INTERNAL "")
    set(${NAME}_DEFINITIONS "-D${LIBFIND_DEFINE}")
  endif ()

  print_library_status(${NAME}
    DETAILS "[${${NAME}_LIBRARY_DIRS}][${${NAME}_LIBRARIES}][${${NAME}_INCLUDE_DIRS}]"
    )

endmacro(libfind_lib_with_pkg_config)
