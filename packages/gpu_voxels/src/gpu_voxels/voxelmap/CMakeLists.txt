# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Sebastian Klemm
# \date    2012-06-06
#
#----------------------------------------------------------------------

#------------- gpu_voxels_voxelmap libaray -----------------------
ICMAKER_SET("gpu_voxels_voxelmap" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  AbstractVoxel.h
  AbstractVoxelMap.h
  BitVoxel.h
  BitVoxelMap.h
  DefaultCollider.h
  ProbabilisticVoxel.h
  ProbVoxelMap.h
  TemplateVoxelMap.h
  VoxelMap.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  kernels/VoxelMapOperations.h
  kernels/VoxelMapOperations.hpp
  kernels/VoxelMapOperations.cu
  AbstractVoxel.h
  AbstractVoxelMap.cu
  AbstractVoxelMap.h
  BitVoxel.h
  BitVoxel.hpp
  BitVoxelMap.h
  BitVoxelMap.hpp
  DefaultCollider.cu
  DefaultCollider.h
  DefaultCollider.hpp
  ProbabilisticVoxel.h
  ProbabilisticVoxel.hpp
  ProbVoxelMap.h
  ProbVoxelMap.hpp
  TemplateVoxelMap.h
  TemplateVoxelMap.hpp
  voxelmap_testcases.cu
  VoxelMap.cu
  VoxelMap.h
  VoxelMap.hpp
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
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
ICMAKER_INSTALL_HEADERS(gpu_voxels/voxelmap)
# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/voxelmap/kernels
  kernels/VoxelMapOperations.h
)
