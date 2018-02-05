# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2015-10-15
#
#----------------------------------------------------------------------

#------------- gpu_voxels_voxel libaray -----------------------
ICMAKER_SET("gpu_voxels_voxel" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  AbstractVoxel.h
  BitVoxel.h
  DefaultCollider.h
  SVCollider.h
  ProbabilisticVoxel.h
  DistanceVoxel.h
  CountingVoxel.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  AbstractVoxel.h
  BitVoxel.h
  BitVoxel.hpp
  DefaultCollider.cu
  DefaultCollider.h
  DefaultCollider.hpp
  SVCollider.h
  SVCollider.hpp
  SVCollider.cu
  ProbabilisticVoxel.h
  ProbabilisticVoxel.hpp
  DistanceVoxel.h
  DistanceVoxel.hpp
  CountingVoxel.h
  CountingVoxel.hpp
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_VOXEL_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_helpers
  gpu_voxels_core
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/voxel)

