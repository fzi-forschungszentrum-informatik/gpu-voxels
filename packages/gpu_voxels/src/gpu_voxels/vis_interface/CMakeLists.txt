# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2012-07-10
#
#----------------------------------------------------------------------

#------------- gpu_voxels_vis_interface libaray -----------------------
ICMAKER_SET("gpu_voxels_vis_interface" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  VisualizerInterface.h
  VisProvider.h
  VisVoxelMap.h
  VisVoxelList.h
  VisTemplateVoxelList.h
  VisTemplateVoxelList.hpp
  VisPrimitiveArray.h
  )

ICMAKER_ADD_SOURCES(
  VisProvider.cpp
  VisVoxelMap.cpp
  VisVoxelList.cpp
  VisTemplateVoxelList.cpp
  VisPrimitiveArray.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_VIS_INTERFACE_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_core
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/vis_interface)
