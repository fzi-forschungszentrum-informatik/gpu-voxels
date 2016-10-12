if ( Osmium_FOUND )
   # in cache already
   SET( Osmium_FIND_QUIETLY TRUE )
endif ()

include(PrintLibraryStatus)
include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(Osmium_PKGCONF osmium)

# Include dir
find_path(Osmium_INCLUDE_DIR
  NAMES osmium.hpp
  PATHS ${Osmium_PKGCONF_INCLUDE_DIRS} "/usr/include/"
)

cmake_policy(SET CMP0017 NEW)

FIND_PACKAGE(Zlib)
FIND_PACKAGE(Pthread)
FIND_PACKAGE(Protobuf)
FIND_PACKAGE(OsmPBF)
FIND_PACKAGE(EXPAT)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this lib depends on.
set(Osmium_PROCESS_INCLUDES Osmium_INCLUDE_DIR OsmPBF_INCLUDE_DIRS)
set(Osmium_PROCESS_LIBS OsmPBF_LIBRARIES PROTOBUF_LIBRARIES EXPAT_LIBRARIES)
libfind_process(Osmium)

PRINT_LIBRARY_STATUS(Osmium
  DETAILS "[${Osmium_LIBRARIES}][${Osmium_INCLUDE_DIRS}]"
)

