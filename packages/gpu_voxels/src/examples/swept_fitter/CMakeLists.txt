# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Klaus Fischnaller
# @date    2016-09-22
#
#----------------------------------------------------------------------


#------------- Example Program swept_fitter ------------
ICMAKER_SET("swept_fitter" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  main.cpp
  swept_fitter/Fitter.cpp
  swept_fitter/GVL.cpp
  swept_fitter/Robot.cpp
  swept_fitter/Trajectory.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_SWEPT_FITTER_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_DEPENDENCIES(EXPORT
  urdfdom
  kdl_parser
  orocos_kdl
  gpu_voxels_urdf_robot
  )


ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  kdl_parser
  gpu_voxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()
