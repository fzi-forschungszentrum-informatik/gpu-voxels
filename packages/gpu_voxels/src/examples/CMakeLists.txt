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
ICMAKER_SET("collisions" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  Collisions.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_COLLISION_)
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
ICMAKER_SET("maps" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  Maps.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_MAPS_)
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
ICMAKER_SET("shift_vs_transform" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  ../gpu_voxels/helpers/Kinect.h
  )

ICMAKER_ADD_SOURCES(
  ShiftVsTransform.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_SHIFT_VS_TRANSFORM_)
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

#------------- Example Program for the usage of height-maps ------------
ICMAKER_SET("heightmap_loader" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  HeightmapLoader.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_HEIGHTMAP_LOADER_)
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
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_SV_VS_ENV_)
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
  PCL
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
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_SANDBOX_)
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
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_URDF_LOADER_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  gpu_voxels_urdf_robot
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program about URDF together with ROS ------------

ICMAKER_SET("urdf_loader_ros_listener" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  URDF_Loader_ROS_listener.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_URDF_LOADER_ROS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  gpu_voxels_urdf_robot
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  ROS
  CUDA
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program about ROS TF interfacing ------------
ICMAKER_SET("tf_interface" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  tfInterface.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_TF_INTERFACE_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  ROS
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
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_ROBOT_VS_ENV_)
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
  PCL
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program for the usage of gpu_voxels ------------
#ICMAKER_SET("ibeo_source" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

#ICMAKER_ADD_HEADERS(
#  IbeoSourceWrapper.h
#  )

#ICMAKER_ADD_SOURCES(
#  IbeoSource.cpp
#  IbeoSourceWrapper.cpp
#  )

#ICMAKER_ADD_CUDA_FILES(
#  )

#ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
#ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_IBEO_SOURCE_)
#ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

#ICMAKER_INTERNAL_DEPENDENCIES(
#  icl_core
#  icl_core_config
#  icl_core_logging
#  icl_sourcesink
#  icl_hardware_ibeo_noapi
#  icl_hardware_ncom
#  icl_gps_types
#  gpu_voxels
#  )

#ICMAKER_EXTERNAL_DEPENDENCIES(
#  CUDA
#  )

#ICMAKER_BUILD_PROGRAM()

#------------- Example Program voxellist sandbox ------------
ICMAKER_SET("voxellist_sandbox" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  VoxelListSandbox.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_VOXELLIST_SANDBOX_)
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



#------------- Example Program voxelmap sandbox ------------
ICMAKER_SET("voxelmap_sandbox" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  VoxelMapSandbox.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_VOXELMAP_SANDBOX_)
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

#------------- Example Program distance_voxel_test ------------
ICMAKER_SET("distance_voxel_test" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})
add_definitions(-DIC_PERFORMANCE_MONITOR)

ICMAKER_ADD_HEADERS(
  )

ICMAKER_ADD_SOURCES(
  DistanceVoxelTest.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_DISTANCE_VOXEL_TEST_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  gpu_voxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  icl_core_performance_monitor
  Boost_SYSTEM
  CUDA
  )

ICMAKER_BUILD_PROGRAM()

#------------- Example Program distance_voxel_test ------------
ICMAKER_SET("distance_kinect_demo" IDE_FOLDER ${EXAMPLES_IDE_FOLDER})
add_definitions(-DIC_PERFORMANCE_MONITOR)

ICMAKER_ADD_HEADERS(
  ../gpu_voxels/helpers/Kinect.h
  )

ICMAKER_ADD_SOURCES(
  DistanceKinectDemo.cpp
  ../gpu_voxels/helpers/Kinect.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_EXAMPLES_DISTANCE_KINECT_DEMO_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  gpu_voxels
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  icl_core_performance_monitor
  Boost_SYSTEM
  CUDA
  OpenNI
  )

ICMAKER_BUILD_PROGRAM()

#------------- adding examples in subdirectories ------------
ADD_SUBDIRECTORY(swept_fitter)
