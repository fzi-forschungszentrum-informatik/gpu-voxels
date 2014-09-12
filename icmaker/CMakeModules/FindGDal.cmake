# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# - Try to find GDal
# Once done, this will define
#
#  GDal_FOUND - system has GDal
#  GDal_INCLUDE_DIR - the GDal include directories
#  GDal_LIBRARY - link these to use GDal

include(PrintLibraryStatus)
include(LibFindMacros)

if ( GDal_FOUND )
   # in cache already
   SET( GDal_FIND_QUIETLY TRUE )
endif ()

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(GDal_PKGCONF gdal)

# Include dir
find_path(GDal_INCLUDE_DIR
  NAMES gdal.h
  PATHS ${GDal_PKGCONF_INCLUDE_DIRS} "/usr/include/gdal" "/opt/local/include"
)

# Finally the library itself
find_library(GDal_LIBRARY
  NAMES gdal gdal1.6.0 gdal1.7.0
  PATHS ${GDal_PKGCONF_LIBRARY_DIRS} "/opt/local/lib"
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(GDal_PROCESS_INCLUDES GDal_INCLUDE_DIR)
set(GDal_PROCESS_LIBS GDal_LIBRARY)
libfind_process(GDal)

PRINT_LIBRARY_STATUS(GDal
  DETAILS "[${GDal_LIBRARIES}][${GDal_INCLUDE_DIRS}]"
)
