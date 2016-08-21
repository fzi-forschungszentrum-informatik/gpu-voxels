# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------

#------------- Example Program for the usage of gpu_voxels ------------
ICMAKER_SET("ibeo_sink_gpuvoxels" IDE_FOLDER ${GPU_CLASSIFICATION_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  sink/IbeoSinkGPUVoxel.h
  filter/IbeoNcomFilter.h
  sink/MetapointcloudSinkGpuvoxel.h
  sink/PointcloudSinkGpuvoxel.h
  PointcloudSourceSinkFilters.h
  filter/IbeoNcomFilterGPU.h
  kernel/ClassificationKernels.h
  SegmentationHelper.h
  GroundPlaneSegmentationHelper.h
  ObjectSegmentationHelper.h
  DecisionTreeClassification.h
  )

ICMAKER_ADD_SOURCES(
  sink/IbeoSinkGPUVoxel.cpp
  filter/IbeoNcomFilter.cpp
  sink/MetapointcloudSinkGpuvoxel.cpp
  sink/PointcloudSinkGpuvoxel.cpp
  SegmentationHelper.cpp
  DecisionTreeClassification.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  filter/IbeoNcomFilterGPU.cu
  kernel/ClassificationKernels.cu
  GroundPlaneSegmentationHelper.cu
  ObjectSegmentationHelper.cu
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  icl_sourcesink
  icl_hardware_ibeo_noapi
  icl_hardware_ncom
  icl_velodyne
  icl_gps_types
  gpu_voxels

  gpu_voxels_urdf_robot
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_LIBRARY()

ICMAKER_INSTALL_HEADER_EXTRAS(../gpu_voxels/robot/kernels
  ../gpu_voxels/robot/kernels/KinematicOperations.h
)

FIND_PACKAGE(classificator_reconfigure)
FIND_PACKAGE(dynamic_reconfigure)
FIND_PACKAGE(pcl_ros)
FIND_PACKAGE(sensor_msgs)

#------------------------------------------------------------------------
ICMAKER_SET("ibeoncomtogpuvoxel" IDE_FOLDER ${GPU_CLASSIFICATION_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  logging/logging_classification.h
  )

ICMAKER_ADD_SOURCES(
  logging/logging_classification.cpp
  IbeoNcomToGpuvoxel.cpp
  )

ICMAKER_ADD_CUDA_FILES(

  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  icl_sourcesink
  icl_hardware_ibeo_noapi
  icl_hardware_ncom
  icl_gps_types
  gpu_voxels
  ibeo_sink_gpuvoxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  ROS
  pcl_ros
  CUDA
  classificator_reconfigure
  dynamic_reconfigure
  )

ICMAKER_BUILD_PROGRAM()


#------------------------------------------------------------------------
ICMAKER_SET("kinectsegmentation" IDE_FOLDER ${GPU_CLASSIFICATION_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  logging/logging_classification.h
  )

ICMAKER_ADD_SOURCES(
  logging/logging_classification.cpp
  KinectExample.cpp
  )

ICMAKER_ADD_CUDA_FILES(

  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  ibeo_sink_gpuvoxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  ROS
  pcl_ros
  sensor_msgs
  #CUDA
  classificator_reconfigure
  dynamic_reconfigure
  )

ICMAKER_BUILD_PROGRAM()

#------------------------------------------------------------------------
ICMAKER_SET("testExample" IDE_FOLDER ${GPU_CLASSIFICATION_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  logging/logging_classification.h
  )

ICMAKER_ADD_SOURCES(
  logging/logging_classification.cpp
  TestExample.cpp
  )

ICMAKER_ADD_CUDA_FILES(

  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  ibeo_sink_gpuvoxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  ROS
  pcl_ros
  sensor_msgs
  #CUDA
  classificator_reconfigure
  dynamic_reconfigure
  )

ICMAKER_BUILD_PROGRAM()


