# - Try to find DlfcnWin32
# Once done, this will define
#
#  DlfcnWin32_FOUND - system has Dlfcn
#  DlfcnWin32_INCLUDE_DIRS - the Dlfcn include directories
#  DlfcnWin32_LIBRARIES - link these to use Dlfcn

IF(DlfcnWin32_FOUND)
   # in cache already
   SET( DlfcnWin32_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

if (DLFCNWIN32_ROOT STREQUAL "" AND "$ENV{DLFCNWIN32_ROOT}" STREQUAL "")
  # Use pkg-config to get hints about paths
  libfind_pkg_check_modules(DlfcnWin32_PKGCONF dlfcn)
else (DLFCNWIN32_ROOT STREQUAL "" AND "$ENV{DLFCNWIN32_ROOT}" STREQUAL "")
  if (DLFCNWIN32_ROOT STREQUAL "")
    set(DLFCNWIN32_ROOT $ENV{DLFCNWIN32_ROOT})
  endif (DLFCNWIN32_ROOT STREQUAL "")
  set(DlfcnWin32_PKGCONF_INCLUDE_DIRS ${DLFCNWIN32_ROOT}/include)
  set(DlfcnWin32_PKGCONF_LIBRARY_DIRS ${DLFCNWIN32_ROOT}/lib)
endif (DLFCNWIN32_ROOT STREQUAL "" AND "$ENV{DLFCNWIN32_ROOT}" STREQUAL "")

# Include dir
find_path(DlfcnWin32_INCLUDE_DIR
  NAMES dlfcn.h
  PATHS ${DlfcnWin32_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(DlfcnWin32_LIBRARY
  NAMES dl
  PATHS ${DlfcnWin32_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(DlfcnWin32_PROCESS_INCLUDES DlfcnWin32_INCLUDE_DIR)
set(DlfcnWin32_PROCESS_LIBS DlfcnWin32_LIBRARY)
libfind_process(DlfcnWin32)

PRINT_LIBRARY_STATUS(DlfcnWin32
  DETAILS "[${DlfcnWin32_LIBRARIES}][${DlfcnWin32_INCLUDE_DIRS}]"
)
