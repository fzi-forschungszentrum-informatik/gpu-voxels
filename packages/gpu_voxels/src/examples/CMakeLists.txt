# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# @date    2014-06-11
#
#----------------------------------------------------------------------


#------------- Example Program for the usage of gpu_voxels ------------
ICMAKER_SET("swept_volume_vs_environment" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  ../gpu_voxels/helpers/Kinect.h
  )

ICMAKER_ADD_SOURCES(
  SweptVolumeVsEnvironment.cpp
  ../gpu_voxels/helpers/Kinect.cpp
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
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  OpenNI
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program sandbox ------------
ICMAKER_SET("sandbox" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  Sandbox.cpp
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
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()


#------------- Example Program about loading URDFs ------------
ICMAKER_SET("urdf_loader" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  URDF_Loader.cpp
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
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program for the usage of gpu_voxels ------------
ICMAKER_SET("robot_vs_environment" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  ../gpu_voxels/helpers/Kinect.h
  )

ICMAKER_ADD_SOURCES(
  RobotVsEnvironment.cpp
  ../gpu_voxels/helpers/Kinect.cpp
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
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  OpenNI
  )

ICMAKER_BUILD_PROGRAM()


#------------- Example Program for the usage of gpu_voxels ------------
ICMAKER_SET("ibeo_source" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  IbeoSourceWrapper.h
  )

ICMAKER_ADD_SOURCES(
  IbeoSource.cpp
  IbeoSourceWrapper.cpp
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
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()
