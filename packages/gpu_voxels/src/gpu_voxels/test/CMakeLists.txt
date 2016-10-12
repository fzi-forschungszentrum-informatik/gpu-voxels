# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Matthias Wagner
# @date    2014-06-13
#
#----------------------------------------------------------------------

#------------- testing_gpu_voxels -----------------------

ICMAKER_SET("test_gpu_voxels_core") # IDE_FOLDER ${ICL_CORE_IDE_FOLDER}

ICMAKER_ADD_HEADERS(
  testing_fixtures.hpp
)

ICMAKER_ADD_SOURCES(
  testing_main.cpp
  ../octree/test/Main_Test.cpp
  ../octree/test/Helper.cpp
  )
ICMAKER_ADD_CUDA_FILES(
  testing_bitvector.cu
  testing_cudaMath.cu
  testing_distance.cu
  testing_octree.cu
  testing_octree_collisions.cu
  testing_pointclouds.cu
  testing_thrust.cu
  testing_voxellist.cu
  testing_voxelmap.cu
  ../octree/test/Tests.cu
  ../octree/test/Tests.h
  )

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels
  gpu_voxels_voxelmap_tests
  )

IF(PCL_FOUND)
  ICMAKER_ADD_SOURCES(
    ../octree/test/ArgumentHandling.cpp
  )
ENDIF(PCL_FOUND)

IF(Boost_FOUND)
  IF(BUILD_SHARED_LIBS)
    ICMAKER_LOCAL_CPPDEFINES("-DBOOST_TEST_DYN_LINK")
  ENDIF(BUILD_SHARED_LIBS)
ENDIF(Boost_FOUND)
ICMAKER_EXTERNAL_DEPENDENCIES(
  Boost_UNIT_TEST_FRAMEWORK
  Boost_PROGRAM_OPTIONS
  )

# removing unknown pragma warnings due to OpenNI spam
#ICMAKER_LOCAL_CPPDEFINES(-DBOOST_TEST_EXPORT_SYMBOLS -Wno-unknown-pragmas)
#ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_BOOST_TEST_)
#ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_BUILD_TEST()

