# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Felix Mauch <mauch@fzi.de>
# \date    2015-05-04
#
#----------------------------------------------------------------------

#------------- gpu_voxels_voxellist libaray -----------------------
ICMAKER_SET("gpu_voxels_voxellist" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  VoxelList.h
  AbstractVoxelList.h
  TemplateVoxelList.h
  BitVoxelList.h
  CountingVoxelList.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  kernels/VoxelListOperations.h
  kernels/VoxelListOperations.hpp
  VoxelList.h
  VoxelList.hpp
  VoxelList.cu
  AbstractVoxelList.h
  AbstractVoxelList.cu
  TemplateVoxelList.h
  TemplateVoxelList.hpp
  BitVoxelList.h
  BitVoxelList.hpp
  CountingVoxelList.h
  CountingVoxelList.hpp
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_VOXELLIST_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_helpers
  gpu_voxels_core
  gpu_voxels_voxel
  gpu_voxels_voxelmap

  CUDA
  Boost_SYSTEM
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/voxellist)
# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/voxellist/kernels
  kernels/VoxelListOperations.h
)
