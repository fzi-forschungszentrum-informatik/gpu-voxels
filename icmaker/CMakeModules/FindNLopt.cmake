# - Try to find the NLopt library (http://ab-initio.mit.edu/wiki/index.php/NLopt)
#
# Once done this will define
#
# NLopt_FOUND         - system has NLopt
# NLopt_INCLUDE_DIR   - the NLopt include directory
# NLopt_LIBRARIES     - Link these to use NLopt
#

# check cache
IF(NLopt_INCLUDE_DIR AND NLopt_LIBRARIES)
  SET(NLopt_FOUND TRUE)
ENDIF(NLopt_INCLUDE_DIR AND NLopt_LIBRARIES)


# not yet cached
IF(NOT NLopt_FOUND)
  FIND_PATH(NLopt_INCLUDE_DIR NAMES nlopt.h)
  FIND_LIBRARY(NLopt_LIBRARIES NAMES nlopt)
  IF(NLopt_INCLUDE_DIR AND NLopt_LIBRARIES)
    SET(NLopt_FOUND TRUE)
    MESSAGE(STATUS "Found NLopt libraries: ${NLopt_LIBRARIES}")
  ENDIF(NLopt_INCLUDE_DIR AND NLopt_LIBRARIES)
ENDIF(NOT NLopt_FOUND)

# did not find NLopt
IF(NOT NLopt_FOUND)
  MESSAGE(STATUS "Could NOT find the NLopt library!")
ENDIF()
