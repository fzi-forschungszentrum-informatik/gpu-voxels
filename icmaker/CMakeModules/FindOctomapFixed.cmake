# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2013-10-07
#
# Try to find OctoMap.  Once done, this will define
#
#  OctomapFixed_FOUND - system has OctoMap
#  OctomapFixed_INCLUDE_DIRS - the OctoMap include directories
#  OctomapFixed_LIBRARY_DIRS - the OctoMap library directories
#  OctomapFixed_LIBRARIES - the OctoMap libraries to link against
#
# The dependency is called OctomapFixed because otherwise the broken
# CMake find script for Octomap that is deployed with some ROS
# versions is found instead, and our fixed find script never gets
# called.
#
#----------------------------------------------------------------------

IF (OctomapFixed_FOUND)
   # in cache already
   SET(OctomapFixed_FIND_QUIETLY TRUE)
ENDIF ()

# library ext. hardcoded to dylib on Apple
if(APPLE)
  set(OctomapFixed_SO_EXT ".dylib")
else(APPLE)
  set(OctomapFixed_SO_EXT ".so")
endif(APPLE)

FIND_LIBRARY(OctomapFixed_LIBRARY
  NAMES
  liboctomap${OctomapFixed_SO_EXT}
  PATHS
  ${OctomapFixed_ROOT}/lib /usr/lib /usr/local/lib /opt/ros/fuerte/lib /opt/ros/groovy/lib /opt/ros/hydro/lib)
IF (NOT OctomapFixed_LIBRARY STREQUAL "OctomapFixed_LIBRARY-NOTFOUND")
  GET_FILENAME_COMPONENT(OctomapFixed_LIBRARY_DIRS ${OctomapFixed_LIBRARY} PATH CACHE)
  GET_FILENAME_COMPONENT(OctomapFixed_ROOT_DIR "${OctomapFixed_LIBRARY_DIRS}" PATH)
  SET(OctomapFixed_INCLUDE_DIRS ${OctomapFixed_ROOT_DIR}/include CACHE INTERNAL "")
  set(OctomapFixed_LIBRARIES
    "${OctomapFixed_LIBRARY_DIRS}/liboctomap${OctomapFixed_SO_EXT};${OctomapFixed_LIBRARY_DIRS}/liboctomath${OctomapFixed_SO_EXT}"
    CACHE INTERNAL ""
    )
  set(OctomapFixed_FOUND True)
  IF (NOT OctomapFixed_FIND_QUIETLY)
    MESSAGE(STATUS "Found Octomap in ${OctomapFixed_LIBRARY}")
  ENDIF ()
ENDIF ()
