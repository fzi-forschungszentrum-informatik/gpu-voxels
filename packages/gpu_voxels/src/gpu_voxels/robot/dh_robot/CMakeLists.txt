# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2012-07-10
#
#----------------------------------------------------------------------


#------------- gpu_voxels_dh_robot libaray -----------------------
ICMAKER_SET("gpu_voxels_dh_robot" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  ../robot_interface.h
  KinematicLink.h
  KinematicChain.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  KinematicLink.h
  KinematicLink.cu
  KinematicChain.h
  KinematicChain.cu
  ../kernels/KinematicOperations.h
  ../kernels/KinematicOperations.cu
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_DH_ROBOT_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_helpers
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/robot/dh_robot)
# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/robot/kernels
  ../kernels/KinematicOperations.h
)
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/robot/
  ../robot_interface.h
)
