# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Florian Drews
# \date    2013-12-01
#
#----------------------------------------------------------------------

ICMAKER_SET("gpu_voxels_octree")

# enable performance monitor
add_definitions(-DIC_PERFORMANCE_MONITOR)

ICMAKER_ADD_HEADERS(
  load_balancer/AbstractLoadBalancer.h
  load_balancer/Extract.h
  load_balancer/Intersect.h
  load_balancer/IntersectVMap.h
  load_balancer/Propagate.h
  DataTypes.h
  EnvironmentNodes.h
  EnvNodesProbabilistic.h
  EnvNodesProbCommon.h
  Morton.h
  Nodes.h
  NTree.h
  NTree.hpp
  NTreeData.h
  Octree.h
  PointCloud.h
  RobotNodes.h
  Sensor.h
  SensorModel.h
  Voxel.h
  DefaultCollider.h
  VisNTree.h
  GvlNTree.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  VisNTree.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  kernels/kernel_common.h
  kernels/kernel_Octree.h
  kernels/kernel_PointCloud.cu
  kernels/kernel_PointCloud.h
  kernels/kernel_traverse.h
  load_balancer/AbstractLoadBalancer.cuh
  load_balancer/AbstractLoadBalancer.h
  load_balancer/Extract.cuh
  load_balancer/Extract.h
  load_balancer/Intersect.cuh
  load_balancer/Intersect.h
  load_balancer/IntersectVMap.cuh
  load_balancer/IntersectVMap.h
  load_balancer/LoadBalancer.cuh
  load_balancer/Propagate.cuh
  load_balancer/Propagate.h
  load_balancer/kernel_config/Extract.cuh
  load_balancer/kernel_config/Intersect.cuh
  load_balancer/kernel_config/IntersectVMap.cuh
  load_balancer/kernel_config/LoadBalance.cuh
  load_balancer/kernel_config/Propagate.cuh
  DataTypes.h
  DefaultCollider.h
  EnvironmentNodes.cu
  EnvironmentNodes.h
  EnvNodesProbabilistic.h
  EnvNodesProbCommon.h
  GvlNTree.h
  GvlNTree.hpp
  Morton.h
  Nodes.h
  NTree.h
  NTree.hpp
  NTreeData.h
  Octree.cu
  Octree.h
  PointCloud.cu
  PointCloud.h
  RobotNodes.cu
  RobotNodes.h
  Sensor.cu
  Sensor.h
  SensorModel.h
  Voxel.h
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_OCTREE_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  gpu_voxels_dh_robot
  gpu_voxels_core
  gpu_voxels_voxel
  gpu_voxels_voxelmap # REMOVE THIS LATER
  gpu_voxels_voxellist # REMOVE THIS LATER
  gpu_voxels_helpers
  gpu_voxels_vis_interface
  icl_core_performance_monitor
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/octree)
# Include files here that are needed when the package is installed.
ADD_SUBDIRECTORY(test)
