# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# - Try to find OGRE (Object-oriented Graphics Rendering Engine)
#

IF(OGRE_FOUND)
   # in cache already
   SET(OGRE_FIND_QUIETLY TRUE)
ENDIF()

include(PrintLibraryStatus)

# OGRE_HOME is used in FindOGRE.cmake to set the base path, if not installed in standard paths
SET(OGRE_HOME "" CACHE PATH "Point to the path where the desired Ogre version can be found." )

IF(OGRE_HOME)
  SET(CMAKE_MODULE_PATH "${OGRE_HOME}/CMake" ${CMAKE_MODULE_PATH})
  include(${OGRE_HOME}/CMake/FindOGRE.cmake)
ELSE()
  # Check for FindOGRE.cmake because it's not deployed by cmake
  FIND_FILE( OGRE_DIST_CMAKE_FILE "FindOGRE.cmake" PATHS "${CMAKE_ROOT}/Modules")
  IF(OGRE_DIST_CMAKE_FILE)
    include(${CMAKE_ROOT}/Modules/FindOGRE.cmake)
  ENDIF()
ENDIF()

PRINT_LIBRARY_STATUS(OGRE
  VERSION "${OGRE_VERSION}"
  DETAILS "[${OGRE_LIBRARY_DIRS}][${OGRE_INCLUDE_DIRS}]"
)
