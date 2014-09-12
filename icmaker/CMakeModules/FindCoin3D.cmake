# - Try to find Coin3D include dirs and libraries
#
# Please see the Documentation for Coin3D in the CMake Manual for details
# This module only forwards to the one included in cmake for compatibility
# reasons.

# Do not change upper case here, as cases are defined in ${CMAKE_ROOT}/Modules/FindCoin3D.cmake
IF(COIN3D_FOUND)
   # in cache already
   SET(COIN3D_FIND_QUIETLY TRUE)
ENDIF()

include(${CMAKE_ROOT}/Modules/FindCoin3D.cmake)

IF (WIN32)
  IF (NOT COIN3D_FOUND)
    if (NOT (COIN3D_ROOT STREQUAL "" AND "$ENV{COIN3D_ROOT}" STREQUAL ""))
      IF (COIN3D_ROOT STREQUAL "")
        SET(COIN3D_ROOT $ENV{COIN3D_ROOT})
      ENDIF (COIN3D_ROOT STREQUAL "")
      
      SET(Coin3D_PKGCONF_INCLUDE_DIRS ${COIN3D_ROOT}/include)
      SET(Coin3D_PKGCONF_LIBRARY_DIRS ${COIN3D_ROOT}/lib)
      
      find_path(COIN3D_INCLUDE_DIRS
        NAMES Inventor/So.h
        PATHS ${Coin3D_PKGCONF_INCLUDE_DIRS}
        )

      find_library(COIN3D_LIBRARY_RELEASE
        NAMES Coin2 Coin3
        PATHS ${Coin3D_PKGCONF_LIBRARY_DIRS}
        )
      find_library(COIN3D_LIBRARY_DEBUG
        NAMES Coin2d Coin3d
        PATHS ${Coin3D_PKGCONF_LIBRARY_DIRS}
        )

      IF (COIN3D_LIBRARY_DEBUG AND COIN3D_LIBRARY_RELEASE)
        SET(COIN3D_LIBRARIES optimized ${COIN3D_LIBRARY_RELEASE}
                             debug ${COIN3D_LIBRARY_DEBUG})
      ELSE (COIN3D_LIBRARY_DEBUG AND COIN3D_LIBRARY_RELEASE)
        IF (COIN3D_LIBRARY_DEBUG)
          SET (COIN3D_LIBRARIES ${COIN3D_LIBRARY_DEBUG})
        ENDIF (COIN3D_LIBRARY_DEBUG)
        IF (COIN3D_LIBRARY_RELEASE)
          SET (COIN3D_LIBRARIES ${COIN3D_LIBRARY_RELEASE})
        ENDIF (COIN3D_LIBRARY_RELEASE)
      ENDIF (COIN3D_LIBRARY_DEBUG AND COIN3D_LIBRARY_RELEASE)

      # handle the QUIETLY and REQUIRED arguments and set COIN3D_FOUND to TRUE if 
      # all listed variables are TRUE
      INCLUDE("${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake")
      FIND_PACKAGE_HANDLE_STANDARD_ARGS(Coin3D DEFAULT_MSG COIN3D_LIBRARIES COIN3D_INCLUDE_DIRS)

      MARK_AS_ADVANCED(COIN3D_INCLUDE_DIRS COIN3D_LIBRARIES)
    ENDIF ()
  ENDIF ()
ENDIF ()

IF (COIN3D_FOUND)
  IF (MSVC)
    IF (BUILD_SHARED_LIBS)
      SET(COIN3D_DEFINITIONS -DCOIN_DLL CACHE INTERNAL "")
    ELSE ()
      SET (COIN3D_DEFINITIONS -DCOIN_NOT_DLL CACHE INTERNAL "")
    ENDIF ()
  ENDIF ()
ENDIF ()
