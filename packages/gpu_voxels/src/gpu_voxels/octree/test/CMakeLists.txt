# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Florian Drews
# \date    2013-12-01
#
#----------------------------------------------------------------------

## ----- programm octree_benchmark -----
#ICMAKER_SET("octree_benchmark")
#
#ICMAKER_ADD_HEADERS(
#  ArgumentHandling.h
#  Helper.h
#  )
#
#ICMAKER_ADD_SOURCES(
#  ArgumentHandling.cpp
#  Helper.cpp
#  Main_Benchmark.cpp
#  )
#
#ICMAKER_ADD_CUDA_FILES(
#  Benchmark_template.h
#  Benchmark.cu
#  Benchmark.h
#  Tests.cu
#  Tests.h
#  )
#
#ICMAKER_LOCAL_CPPDEFINES(-DICL_ENVIRONMENT_GPU_EXPORT_SYMBOLS -Wno-unknown-pragmas)
#ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_OCTREE_BENCH_)
#ICMAKER_INCLUDE_DIRECTORIES(${ICL_ENVIRONMENT_INCLUDE_DIRS})
#ICMAKER_INCLUDE_DIRECTORIES(../)
#ICMAKER_INCLUDE_DIRECTORIES(../../)
#ICMAKER_INCLUDE_DIRECTORIES(../cub)
#
#
#ICMAKER_INTERNAL_DEPENDENCIES(
#  icl_environment_gpu_octree
#  )
#
#ICMAKER_EXTERNAL_DEPENDENCIES(
#  CUDA
#  PCL
#  )
#
#ICMAKER_BUILD_PROGRAM()
#







## ----- programm octree_test -----
#ICMAKER_SET("octree_test")

#ICMAKER_ADD_HEADERS(
#  ArgumentHandling.h
#  Helper.h
#  )

#ICMAKER_ADD_SOURCES(
#  ArgumentHandling.cpp
#  Main_Test.cpp
#  Helper.cpp
#  )

#ICMAKER_ADD_CUDA_FILES(
#  Tests.cu
#  Tests.h
#  )

#ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
#ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_OCTREE_TEST_)
#ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})
#ICMAKER_INCLUDE_DIRECTORIES(../cub)


#ICMAKER_INTERNAL_DEPENDENCIES(
#  gpu_voxels_octree
#  )

#ICMAKER_EXTERNAL_DEPENDENCIES(
#  CUDA
#  PCL
#  )

#ICMAKER_BUILD_PROGRAM()


# ----- program octree_provider -----
ICMAKER_SET("octree_provider")
# enable performance monitor
add_definitions(-DIC_PERFORMANCE_MONITOR)

ICMAKER_ADD_HEADERS(
  ArgumentHandling.h
  Kinect.h
  OctomapProvider.h
  Provider.h
  SensorData.h
  Helper.h
  NTreeProvider.h
  )

ICMAKER_ADD_SOURCES(
  ArgumentHandling.cpp
  Kinect.cpp
  Main_NTreeProvider.cpp
  OctomapProvider.cpp
  Provider.cpp
  Helper.cpp
  VoxelMapProvider.cpp
  NTreeProvider.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_OCTREE_PROVIDER_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})
ICMAKER_INCLUDE_DIRECTORIES(../cub)
ICMAKER_INCLUDE_DIRECTORIES(../octomap/octomap/include/)


ICMAKER_INTERNAL_DEPENDENCIES(
  gpu_voxels_octree
  gpu_voxels_voxel
  gpu_voxels_voxelmap
  icl_core
  icl_core_performance_monitor
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  PCL
  Octomap
  ROS
  )

# LibRT is needed for Boost Interprocess on POSIX systems
IF(UNIX AND NOT APPLE)
  ICMAKER_EXTERNAL_DEPENDENCIES(
    LibRt
    )
ENDIF(UNIX AND NOT APPLE)


ICMAKER_BUILD_PROGRAM()




# ----- program octree_provider -----
ICMAKER_SET("octree_bench")

# enable performance monitor
add_definitions(-DIC_PERFORMANCE_MONITOR)

ICMAKER_ADD_HEADERS(
  ArgumentHandling.h
  Kinect.h
  OctomapProvider.h
  Provider.h
  SensorData.h
  Helper.h
  NTreeProvider.h
  )

ICMAKER_ADD_SOURCES(
  ArgumentHandling.cpp
  Kinect.cpp
  Main_Bench.cpp
  OctomapProvider.cpp
  Provider.cpp
  Helper.cpp
  NTreeProvider.cpp
  VoxelMapProvider.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_OCTREE_BENCH_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})
ICMAKER_INCLUDE_DIRECTORIES(${ICL_HARDWARE_INCLUDE_DIRS})
ICMAKER_INCLUDE_DIRECTORIES(../cub)
ICMAKER_INCLUDE_DIRECTORIES(../octomap/octomap/include/)


ICMAKER_INTERNAL_DEPENDENCIES(
  gpu_voxels_voxel
  gpu_voxels_octree
  gpu_voxels_voxelmap
  icl_core
  icl_core_performance_monitor
  ROS
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  CUDA
  PCL
  Octomap
  Boost_DATE_TIME
  )

# LibRT is needed for Boost Interprocess on POSIX systems
IF(UNIX AND NOT APPLE)
  ICMAKER_EXTERNAL_DEPENDENCIES(
    LibRt
    )
ENDIF(UNIX AND NOT APPLE)

ICMAKER_BUILD_PROGRAM()

