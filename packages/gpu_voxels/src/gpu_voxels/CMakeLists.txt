# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Sebastian Klemm
# \date    2012-06-06
#
#----------------------------------------------------------------------

# Used in IcWorkspace's doc/doxyfile.in to set the title of all pages.
SET(IC_WORKSPACE_DOXYGEN_PROJECT_NAME "GPU-Voxels" CACHE INTERNAL "")
# Used in IcWorkspace's doc/doxyfile.in to overwrite configs
# from individual CMakeLists.txt files
#SET(IC_WORKSPACE_DOXYGEN_CONFIG_OVERRIDES "gpuvoxels" CACHE INTERNAL "")

#------------- anounce basic libs ----------------------------
ADD_SUBDIRECTORY(logging)
ADD_SUBDIRECTORY(helpers)
ADD_SUBDIRECTORY(primitive_array) #could later be splitted up into basic and specific type

#------------- gpu_voxels_core libaray -----------------------
ICMAKER_SET("gpu_voxels_core" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  GpuVoxelsMap.h
  ManagedMap.h
  ManagedPrimitiveArray.h
  )

ICMAKER_ADD_SOURCES(
  GpuVoxelsMap.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  octree/DataTypes.h
  )



# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_CORE_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_helpers
  gpu_voxels_primitive_array
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
  Boost_THREAD
  Boost_SYSTEM
  Boost_CHRONO
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels)

#------------- anounce high level libs and tests ------------------
ADD_SUBDIRECTORY(robot)
ADD_SUBDIRECTORY(voxel)
ADD_SUBDIRECTORY(voxelmap)
ADD_SUBDIRECTORY(voxellist)
ADD_SUBDIRECTORY(vis_interface)
ADD_SUBDIRECTORY(octree)
ADD_SUBDIRECTORY(helpers_highlevel)

#------------- gpu_voxels library ---------------------------------
# This library resembles the high level interface to link against.
#------------------------------------------------------------------
ICMAKER_SET("gpu_voxels" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  GpuVoxels.h
  )

ICMAKER_ADD_SOURCES(
  GpuVoxels.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGVL_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})


ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_core
  gpu_voxels_dh_robot
  gpu_voxels_voxelmap
  gpu_voxels_voxellist
  gpu_voxels_octree
  gpu_voxels_primitive_array
  gpu_voxels_vis_interface
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
  )


ICMAKER_DEPENDENCIES( OPTIONAL
  gpu_voxels_urdf_robot
)

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels)

ADD_SUBDIRECTORY(test)

