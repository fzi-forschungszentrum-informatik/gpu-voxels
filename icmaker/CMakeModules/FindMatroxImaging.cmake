# - Try to find MatroxImaging
# Once done, this will define
#
#  MatroxImaging_FOUND - system has MatroxImaging
#  MatroxImaging_INCLUDE_DIR - the MatroxImaging include directories
#  MatroxImaging_LIBRARY - link these to use MatroxImaging

include(PrintLibraryStatus)
include(LibFindMacros)

if ( MatroxImaging_FOUND )
   # in cache already
   SET( MatroxImaging_FIND_QUIETLY TRUE )
endif ()

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(MatroxImaging_PKGCONF matrox_imaging)

# Include dir
find_path(MatroxImaging_INCLUDE_DIR
  NAMES mil.h
  PATHS ${MatroxImaging_PKGCONF_INCLUDE_DIRS} "/opt/matrox_imaging/mil/include"
)

# Finally the library itself
find_library(MatroxImaging_LIBRARY
  NAMES mil mil.so.10 mil.so.10.00.2378
  PATHS ${MatroxImaging_PKGCONF_LIBRARY_DIRS} "/opt/matrox_imaging/mil/lib"
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(MatroxImaging_PROCESS_INCLUDES MatroxImaging_INCLUDE_DIR)
set(MatroxImaging_PROCESS_LIBS MatroxImaging_LIBRARY)
libfind_process(MatroxImaging)

PRINT_LIBRARY_STATUS(MatroxImaging
  DETAILS "[${MatroxImaging_LIBRARIES}][${MatroxImaging_INCLUDE_DIRS}]"
)
