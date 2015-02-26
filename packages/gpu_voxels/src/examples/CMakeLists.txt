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
  Boost_SYSTEM
  PCL
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program sandbox ------------
ICMAKER_SET("sandbox" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  sandbox.cpp
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
  Boost_SYSTEM
  PCL
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
  Boost_SYSTEM
  PCL
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()

#------------- voxelmap_provider_test -----------------------

ICMAKER_SET("voxelmap_provider_test")

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  voxelmap_provider_test.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  )

FIND_PACKAGE(GLM)

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  OpenNI
  Boost_SYSTEM
  PCL
  OpenNI
  OpenGL
  GLUT
  GLM
  )

# LibRT is needed for Boost Interprocess on POSIX systems
IF(UNIX AND NOT APPLE)
  FIND_PACKAGE(LibRt)
  ICMAKER_EXTERNAL_DEPENDENCIES(
    LibRt
    )
ENDIF(UNIX AND NOT APPLE)

ICMAKER_BUILD_PROGRAM()

#------------- Example Program for the usage of an octree environment map ------------
ICMAKER_SET("octree_environment_map" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  OctreeEnvironmentMap.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  gpu_voxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
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
  PCL
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()
