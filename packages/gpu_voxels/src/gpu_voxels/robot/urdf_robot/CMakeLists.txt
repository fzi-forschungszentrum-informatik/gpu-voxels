# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2012-07-10
#
#----------------------------------------------------------------------

#------------- gpu_voxels_urdf_robot libaray -----------------------
IF(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)
  ICMAKER_SET("gpu_voxels_urdf_robot" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

  ICMAKER_GLOBAL_CPPDEFINES(-D_BUILD_GVL_WITH_URDF_SUPPORT_)

  ICMAKER_ADD_HEADERS(
      node.h
      robot.h
      robot_joint.h
      robot_link.h
      urdf_robot.h
      robot_to_gpu.h
    )

  ICMAKER_ADD_SOURCES(
      ../robot_interface.h
      node.cpp
      robot.cpp
      robot_joint.cpp
      robot_link.cpp
      urdf_robot.cpp
    )

  ICMAKER_ADD_CUDA_FILES(
    robot_to_gpu.h
    robot_to_gpu.cu
    ../kernels/KinematicOperations.h
    ../kernels/KinematicOperations.cu
    )

  # removing unknown pragma warnings due to OpenNI spam
  ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
  ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_URDF_ROBOT_)
  ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

  ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging
    gpu_voxels_helpers
    )

  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
    urdfdom
    kdl_parser
    orocos_kdl
    Boost_FILESYSTEM
    )


  ICMAKER_BUILD_LIBRARY()
  ICMAKER_INSTALL_HEADERS(gpu_voxels/robot/urdf_robot)
  # Include files here that are needed when the package is installed.
  ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/robot/kernels
    ../kernels/KinematicOperations.h
  )
  ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/robot/
    ../robot_interface.h
  )

ENDIF(urdfdom_FOUND AND orocos_kdl_FOUND AND kdl_parser_FOUND)
