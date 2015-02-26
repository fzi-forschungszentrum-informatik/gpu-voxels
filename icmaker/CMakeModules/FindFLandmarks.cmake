# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# - Try to find FLandmarks
# Once done, this will define
#
#  FLandmarks_FOUND - system has GDal
#  FLandmarks_INCLUDE_DIR - the GDal include directories
#  FLandmarks_LIBRARY - link these to use GDal

include(PrintLibraryStatus)
include(LibFindMacros)

if ( FLandmarks_FOUND )
   # in cache already
   SET( FLandmarks_FIND_QUIETLY TRUE )
endif ()

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(FLandmarks_PKGCONF flandmarks)

# Include dir
find_path(FLandmarks_INCLUDE_DIR
  NAMES flandmark_detector.h
  PATHS ${FLandmarks_PKGCONF_INCLUDE_DIRS} "/usr/include/flandmarks" "/opt/local/include"
)

# Finally the library itself
find_library(FLandmarks_LIBRARY
  NAMES flandmark_shared
  PATHS ${FLandmarks_PKGCONF_LIBRARY_DIRS} "/opt/local/lib"
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(FLandmarks_PROCESS_INCLUDES FLandmarks_INCLUDE_DIR)
set(FLandmarks_PROCESS_LIBS FLandmarks_LIBRARY)
libfind_process(FLandmarks)

PRINT_LIBRARY_STATUS(FLandmarks
  DETAILS "[${FLandmarks_LIBRARIES}][${FLandmarks_INCLUDE_DIRS}]"
)
