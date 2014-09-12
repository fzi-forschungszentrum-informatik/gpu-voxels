# - Try to find scip library
# Once done this will define
#  SCIP_FOUND - System has libscipopt
#  SCIP_INCLUDE_DIRS - The libscipopt include directories
#  SCIP_LIBRARIES - link these to use libscipopt

include(PrintLibraryStatus)
include(LibFindMacros)

SET(SCIP_SEARCH_INCLUDE_PATHS /usr/include/scip;/usr/local/include/scip)
SET(SCIP_SEARCH_LIB_PATHS /usr/lib;/usr/local/lib)

find_path(SCIP_INCLUDE_DIR
  NAMES scip.h
  PATHS ${SCIP_SEARCH_INCLUDE_PATHS}
)

find_library(SCIP_LIBRARY
  NAMES libscipopt.so.3.1.0
  PATHS ${SCIP_SEARCH_LIB_PATHS}
)

set(SCIP_PROCESS_INCLUDES SCIP_INCLUDE_DIR)
set(SCIP_PROCESS_LIBS SCIP_LIBRARY)

libfind_process(SCIP)

PRINT_LIBRARY_STATUS(SCIP
  DETAILS "[${SCIP_LIBRARIES}][${SCIP_INCLUDE_DIRS}]"
)

