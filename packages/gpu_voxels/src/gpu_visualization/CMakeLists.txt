# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2014-06-11
#
#----------------------------------------------------------------------

# ----- library gpu_voxels_visualization_core -----
ICMAKER_SET("gpu_voxels_visualization_core")

ICMAKER_ADD_HEADERS(
  Utils.h
  shader.h
  Camera.h
  XMLInterpreter.h
  VisualizerContext.h
  Cuboid.h
  Sphere.h
  visualizerDefines.h
  Visualizer.h
  kernels/VoxelMapVisualizerOperations.h
  logging/logging_visualization.h
  SharedMemoryManager.h
  SharedMemoryManagerOctrees.h
  SharedMemoryManagerVoxelMaps.h
  SharedMemoryManagerVoxelLists.h
  SharedMemoryManagerPrimitiveArrays.h
  SharedMemoryManagerVisualizer.h
  )

ICMAKER_ADD_SOURCES(
  Utils.cpp
  shader.cpp
  Camera.cpp
  XMLInterpreter.cpp
  logging/logging_visualization.cpp
  SharedMemoryManagerOctrees.cpp
  SharedMemoryManagerVoxelMaps.cpp
  SharedMemoryManagerVoxelLists.cpp
  SharedMemoryManagerPrimitiveArrays.cpp
  SharedMemoryManagerVisualizer.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  Visualizer.cu
  kernels/VoxelMapVisualizerOperations.h
  kernels/VoxelMapVisualizerOperations.cu
  VisualizerContext.h
  )


ICMAKER_LOCAL_CPPDEFINES(-DGPU_VIS_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VIS_CORE_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_logging
  gpu_voxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
  OpenGL
  GLUT
  GLEW
  GLM
  Boost_FILESYSTEM
  Boost_SYSTEM
  Boost_THREAD
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels_visualization)

#------------- gpu_voxels_visualizer -----------------------

ICMAKER_SET("gpu_voxels_visualizer")

ICMAKER_ADD_HEADERS(
  gpu_voxels_visualizer.h
  )

ICMAKER_ADD_SOURCES(
  gpu_voxels_visualizer.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VIS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VIS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_visualization_core
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )


# LibRT is needed for Boost Interprocess on POSIX systems
IF(UNIX AND NOT APPLE)
  ICMAKER_EXTERNAL_DEPENDENCIES(
    LibRt
    )
ENDIF(UNIX AND NOT APPLE)

ICMAKER_BUILD_PROGRAM()
