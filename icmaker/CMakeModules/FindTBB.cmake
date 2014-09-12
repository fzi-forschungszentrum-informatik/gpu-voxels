# - Try to find TBB
# Once done, this will define
#
#  TBB_FOUND - system has TBB
#  TBB_INCLUDE_DIRS - the TBB include directories
#  TBB_LIBRARIES - link these to use TBB

IF(TBB_FOUND)
   # in cache already
   SET( TBB_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

if (TBB_ROOT STREQUAL "" AND "$ENV{TBB_ROOT}" STREQUAL "")
  # Use pkg-config to get hints about paths
  libfind_pkg_check_modules(TBB_PKGCONF tbb)
else (TBB_ROOT STREQUAL "" AND "$ENV{TBB_ROOT}" STREQUAL "")
  if (TBB_ROOT STREQUAL "")
    set(TBB_ROOT $ENV{TBB_ROOT})
  endif (TBB_ROOT STREQUAL "")
  set(TBB_PKGCONF_INCLUDE_DIRS ${TBB_ROOT}/include)
  set(TBB_PKGCONF_LIBRARY_DIRS ${TBB_ROOT}/lib)
endif (TBB_ROOT STREQUAL "" AND "$ENV{TBB_ROOT}" STREQUAL "")

# Include dir
find_path(TBB_INCLUDE_DIR
  NAMES tbb/tbb.h
  PATHS ${TBB_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(TBB_LIBRARY
  NAMES tbb
  PATHS ${TBB_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(TBB_PROCESS_INCLUDES TBB_INCLUDE_DIR)
set(TBB_PROCESS_LIBS TBB_LIBRARY)
libfind_process(TBB)

PRINT_LIBRARY_STATUS(TBB
  DETAILS "[${TBB_LIBRARIES}][${TBB_INCLUDE_DIRS}]"
)
