# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# @date    2017-04-10
#
# Here all the helpers should be collected, which requrire high-level
# functionality of GPU-Voxels.
#----------------------------------------------------------------------


#------------- distance_map_converter libaray -----------------------
ICMAKER_SET("gpu_voxels_distance_map_converter" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  DistanceMapConverter.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  DistanceMapConverter.cu
  DistanceMapConverter.h
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_DIST_MAP_CONV_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_voxelmap
  gpu_voxels_voxellist
  gpu_voxels_voxel
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
)


ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/helpers_highlevel)
