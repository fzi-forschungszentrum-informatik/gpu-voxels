# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

include(${CMAKE_ROOT}/Modules/FindOpenMP.cmake)

IF(OPENMP_FOUND)
  SET(OpenMP_DEFINITIONS ${OpenMP_CXX_FLAGS})
  IF(NOT WIN32)
    SET(OpenMP_LIBRARIES ${OpenMP_CXX_FLAGS})
  ENDIF()
ENDIF()
