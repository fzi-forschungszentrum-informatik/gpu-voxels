# - Try to find Antlr3
# Once done, this will define
#
#  Antlr3_FOUND - system has Antlr3
#  Antlr3_INCLUDE_DIRS - the Antlr3 include directories
#  Antlr3_LIBRARIES - link these to use Antlr3

include(LibFindMacros)

if (NOT ANTLR3_ROOT STREQUAL "" OR NOT "$ENV{ANTLR3_ROOT}" STREQUAL "")
  if (ANTLR3_ROOT STREQUAL "")
    set(ANTLR3_ROOT $ENV{ANTLR3_ROOT})
  endif (ANTLR3_ROOT STREQUAL "")
  SET (Antlr3_INCLUDE_SEARCH_DIRS ${ANTLR3_ROOT}/include)
  SET (Antlr3_LIB_SEARCH_DIRS ${ANTLR3_ROOT}/lib)
endif (NOT ANTLR3_ROOT STREQUAL "" OR NOT "$ENV{ANTLR3_ROOT}" STREQUAL "")

find_path(Antlr3_INCLUDE_DIR
  NAMES antlr3.h
  PATHS ${Antlr3_INCLUDE_SEARCH_DIRS}
)
find_library(Antlr3_LIBRARY
  NAMES antlr3c libantlr3c
  PATHS ${Antlr3_LIB_SEARCH_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Antlr3_PROCESS_INCLUDES Antlr3_INCLUDE_DIR)
set(Antlr3_PROCESS_LIBS Antlr3_LIBRARY)
libfind_process(Antlr3)
