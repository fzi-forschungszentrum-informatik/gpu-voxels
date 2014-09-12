# - Try to find Mlpack
# Once done, this will define
#
#  Mlpack_FOUND - system has Mlpack
#  Mlpack_INCLUDE_DIRS - the Mlpack include directories
#  Mlpack_LIBRARIES - link these to use Mlpack

IF(Mlpack_FOUND)
   # in cache already
   SET( Mlpack_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Dependencies
# libfind_package(Mlpack Freetype)

if (MLPACK_ROOT STREQUAL "" AND "$ENV{MLPACK_ROOT}" STREQUAL "")
  # Use pkg-config to get hints about paths
  libfind_pkg_check_modules(Mlpack_PKGCONF mlpack)
else (MLPACK_ROOT STREQUAL "" AND "$ENV{MLPACK_ROOT}" STREQUAL "")
  if (MLPACK_ROOT STREQUAL "")
    set(MLPACK_ROOT $ENV{MLPACK_ROOT})
  endif (MLPACK_ROOT STREQUAL "")
  set(Mlpack_PKGCONF_INCLUDE_DIRS ${MLPACK_ROOT}/include)
  set(Mlpack_PKGCONF_LIBRARY_DIRS ${MLPACK_ROOT}/lib)
endif (MLPACK_ROOT STREQUAL "" AND "$ENV{MLPACK_ROOT}" STREQUAL "")

# Include dir
find_path(Mlpack_INCLUDE_DIR
  NAMES mlpack/core.hpp
  PATHS ${Mlpack_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(Mlpack_LIBRARY
  NAMES mlpack
  PATHS ${Mlpack_PKGCONF_LIBRARY_DIRS}
)

# Include dir
find_path(xml2_INCLUDE_DIR
  NAMES libxml/parser.h
  PATHS /usr/include/libxml2
)

find_library(xml2_LIBRARY
  NAMES xml2
  PATHS ${Mlpack_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Mlpack_PROCESS_INCLUDES Mlpack_INCLUDE_DIR xml2_INCLUDE_DIR)
set(Mlpack_PROCESS_LIBS Mlpack_LIBRARY xml2_LIBRARY)
libfind_process(Mlpack)

PRINT_LIBRARY_STATUS(Mlpack
  DETAILS "[${Mlpack_LIBRARIES}][${Mlpack_INCLUDE_DIRS}]"
)
