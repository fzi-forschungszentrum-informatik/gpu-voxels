# - Try to find SimD-cxx
# Once done, this will define
#
#  SimDcxx_FOUND - system has SimD-cxx
#  SimDcxx_INCLUDE_DIRS - the SimD-cxx include directories
#  SimDcxx_LIBRARIES - link these to use SimD-cxx

IF( SimDcxx_FOUND )
   # in cache already
   SET( SimDcxx_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

if (NOT SIMDCXX_ROOT STREQUAL "" OR NOT "$ENV{SIMDCXX_ROOT}" STREQUAL "")
  if (SIMDCXX_ROOT STREQUAL "")
    set(SIMDCXX_ROOT $ENV{SIMDCXX_ROOT})
  endif (SIMDCXX_ROOT STREQUAL "")
  SET (SimDcxx_INCLUDE_SEARCH_DIRS ${SIMDCXX_ROOT}/include)
  SET (SimDcxx_LIB_SEARCH_DIRS ${SIMDCXX_ROOT}/lib)
endif (NOT SIMDCXX_ROOT STREQUAL "" OR NOT "$ENV{SIMDCXX_ROOT}" STREQUAL "")

# Include dir
find_path(SimDcxx_INCLUDE_DIR
  NAMES dds/dds.hpp
  PATHS ${SimDcxx_INCLUDE_SEARCH_DIRS}
)

# Finally the library itself
find_library(SimDcxx_LIBRARY
  NAMES SimD
  PATHS ${SimDcxx_LIB_SEARCH_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries that this lib depends on.
set(SimDcxx_PROCESS_INCLUDES SimDcxx_INCLUDE_DIR)
set(SimDcxx_PROCESS_LIBS SimDcxx_LIBRARY)
libfind_process(SimDcxx)

PRINT_LIBRARY_STATUS(SimDcxx
  DETAILS "[${SimDcxx_LIBRARIES}][${SimDcxx_INCLUDE_DIRS}]"
)
