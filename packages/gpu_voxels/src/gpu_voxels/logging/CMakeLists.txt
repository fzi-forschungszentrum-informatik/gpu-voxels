# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2012-07-10
#
#----------------------------------------------------------------------


#------------- gpu_voxels_logging libaray -----------------------
ICMAKER_SET("gpu_voxels_logging" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  logging_gpu_voxels.h
  logging_gpu_voxels_helpers.h
  logging_cuda.h
  logging_voxelmap.h
  logging_voxellist.h
  logging_primitive_array.h
  logging_robot.h
  logging_octree.h
  )

ICMAKER_ADD_SOURCES(
  logging_gpu_voxels.cpp
  logging_gpu_voxels_helpers.cpp
  logging_cuda.cpp
  logging_voxelmap.cpp
  logging_voxellist.cpp
  logging_primitive_array.cpp
  logging_robot.cpp
  logging_octree.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_LOGGING_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/logging)
