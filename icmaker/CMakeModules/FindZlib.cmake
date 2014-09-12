# - Try to find Zlib
# Once done, this will define
#
#  Zlib_FOUND - system has Zlib
#  Zlib_INCLUDE_DIRS - the Zlib include directories
#  Zlib_LIBRARIES - link these to use Zlib

IF(Zlib_FOUND)
   # in cache already
   SET( Zlib_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Dependencies
# libfind_package(Zlib Freetype)

if (ZLIB_ROOT STREQUAL "" AND "$ENV{ZLIB_ROOT}" STREQUAL "")
  # Use pkg-config to get hints about paths
  libfind_pkg_check_modules(Zlib_PKGCONF zlib)
else (ZLIB_ROOT STREQUAL "" AND "$ENV{ZLIB_ROOT}" STREQUAL "")
  if (ZLIB_ROOT STREQUAL "")
    set(ZLIB_ROOT $ENV{ZLIB_ROOT})
  endif (ZLIB_ROOT STREQUAL "")
  set(Zlib_PKGCONF_INCLUDE_DIRS ${ZLIB_ROOT}/include)
  set(Zlib_PKGCONF_LIBRARY_DIRS ${ZLIB_ROOT}/lib)
endif (ZLIB_ROOT STREQUAL "" AND "$ENV{ZLIB_ROOT}" STREQUAL "")

# Include dir
find_path(Zlib_INCLUDE_DIR
  NAMES zlib.h
  PATHS ${Zlib_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(Zlib_LIBRARY
  NAMES z zlib
  PATHS ${Zlib_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Zlib_PROCESS_INCLUDES Zlib_INCLUDE_DIR)
set(Zlib_PROCESS_LIBS Zlib_LIBRARY)
libfind_process(Zlib)

PRINT_LIBRARY_STATUS(Zlib
  DETAILS "[${Zlib_LIBRARIES}][${Zlib_INCLUDE_DIRS}]"
)
