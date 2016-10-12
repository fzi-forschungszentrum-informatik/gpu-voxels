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

#SET(ICMAKER_CUDA_CPPDEFINES ${ICMAKER_CUDA_CPPDEFINES} --ptxas-options=-v)
add_definitions(-DIC_PERFORMANCE_MONITOR)

ICMAKER_ADD_HEADERS(
  AbstractVoxelMap.h
  BitVoxelMap.h
  ProbVoxelMap.h
  TemplateVoxelMap.h
  VoxelMap.h
  DistanceVoxelMap.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  kernels/VoxelMapOperations.h
  kernels/VoxelMapOperationsPBA.h
  kernels/VoxelMapOperations.hpp
  kernels/VoxelMapOperationsPBA.hpp
  kernels/VoxelMapOperations.cu
  AbstractVoxelMap.cu
  AbstractVoxelMap.h
  BitVoxelMap.h
  BitVoxelMap.hpp
  ProbVoxelMap.h
  ProbVoxelMap.hpp
  TemplateVoxelMap.h
  TemplateVoxelMap.hpp
  VoxelMap.cu
  VoxelMap.h
  VoxelMap.hpp
  DistanceVoxelMap.h
  DistanceVoxelMap.hpp
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_VOXELMAP_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  icl_core_performance_monitor
  gpu_voxels_logging
  gpu_voxels_helpers
  gpu_voxels_core
  gpu_voxels_voxel
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )
  
ICMAKER_INSTALL_HEADERS(gpu_voxels/voxelmap)

# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/voxelmap/kernels
  kernels/VoxelMapOperations.h
  kernels/VoxelMapOperationsPBA.h
)
  
ICMAKER_BUILD_LIBRARY()

#------------- gpu_voxels_voxelmap_tests libaray -----------------------
ICMAKER_SET("gpu_voxels_voxelmap_tests" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  Tests.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  kernels/VoxelMapOperations.h
  kernels/VoxelMapOperationsPBA.h
  kernels/VoxelMapOperations.hpp
  kernels/VoxelMapOperationsPBA.hpp
  kernels/VoxelMapOperations.cu
  kernels/VoxelMapTests.h
  kernels/VoxelMapTests.hpp
  kernels/VoxelMapTests.cu
  Tests.hpp
  Tests.cu
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_VOXELMAP_TESTS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_helpers
  gpu_voxels_core
  gpu_voxels_voxel
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_INSTALL_HEADERS(gpu_voxels/voxelmap)

# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/voxelmap/kernels
  kernels/VoxelMapTests.h
)

  
ICMAKER_BUILD_LIBRARY()
