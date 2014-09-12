# - Find the libfreenect includes and library
# This module defines
#  FREENECT_INCLUDE_DIR, path to libfreenect/libfreenect.h, etc.
#  FREENECT_LIBRARIES, the libraries required to use FREENECT.
#  FREENECT_FOUND, If false, do not try to use FREENECT.
# also defined, but not for general use are
#  FREENECT_freenect_LIBRARY, where to find the FREENECT library.

FIND_PATH(FREENECT_INCLUDE_DIR libfreenect.h
  /usr/include/libfreenect
  /usr/local/include
  /usr/local/include/libfreenect
  /usr/include
)

FIND_LIBRARY(FREENECT_freenect_LIBRARY freenect
  /usr/local/lib
  /usr/lib
)

MARK_AS_ADVANCED(
  FREENECT_INCLUDE_DIR
  FREENECT_freenect_LIBRARY)

SET( FREENECT_FOUND "NO" )
IF(FREENECT_INCLUDE_DIR)
  IF(FREENECT_freenect_LIBRARY)
    SET( FREENECT_FOUND "YES" )
    SET( FREENECT_LIBRARIES
      ${FREENECT_freenect_LIBRARY})
  ENDIF(FREENECT_freenect_LIBRARY)
ENDIF(FREENECT_INCLUDE_DIR)

IF(FREENECT_FOUND)
  MESSAGE(STATUS "Found freenect library")
ELSE(FREENECT_FOUND)
  IF(FREENECT_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find libfreenect 
-- please give some paths to CMake or make sure libfreenect is installed in your system")
  ELSE(FREENECT_FIND_REQUIRED)
    MESSAGE(STATUS "Could not find libfreenect 
-- please give some paths to CMake or make sure libfreenect is installed in your system")
  ENDIF(FREENECT_FIND_REQUIRED)
ENDIF(FREENECT_FOUND)
