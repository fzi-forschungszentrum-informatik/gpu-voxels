# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
#
# \author  Andreas Hermann
# \date    2012-07-10
#
#----------------------------------------------------------------------


#------------- gpu_voxels_helpers libaray -----------------------
ICMAKER_SET("gpu_voxels_helpers" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  GlobalLock.h
  cuda_vectors.h
  cuda_matrices.h
  cuda_datatypes.h
  common_defines.h
  cuda_handling.h
  cuda_handling.hpp
  HeightMapLoader.h
  PointcloudFileHandler.h
  BinvoxFileReader.h
  FileReaderInterface.h
  XyzFileReader.h
  PointCloud.h
  MetaPointCloud.h
  BitVector.h
  CompileIssues.h
  MathHelpers.h
  GeometryGeneration.h
  CollisionInterfaces.h
  stb_image.h
  )

ICMAKER_ADD_SOURCES(
  GlobalLock.cpp
  cuda_datatypes.cpp
  HeightMapLoader.cpp
  PointcloudFileHandler.cpp
  BinvoxFileReader.cpp
  XyzFileReader.cpp
  MathHelpers.cpp
  GeometryGeneration.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  CompileIssues.h
  cuda_handling.h
  cuda_handling.hpp
  cuda_handling.cu
  PointCloud.h
  PointCloud.cu
  MetaPointCloud.cu
  kernels/MetaPointCloudOperations.h
  kernels/MetaPointCloudOperations.cu
  kernels/HelperOperations.h
  kernels/HelperOperations.cu
  BitVector.h
  )

IF(PCL_FOUND)
  ICMAKER_GLOBAL_CPPDEFINES(-D_BUILD_GVL_WITH_PCL_SUPPORT_)
  ICMAKER_ADD_HEADERS(
    PcdFileReader.h
  )
  ICMAKER_ADD_SOURCES(
    PcdFileReader.cpp
  )
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
    PCL
  )

  IF(OPENNI_FOUND)
    ICMAKER_GLOBAL_CPPDEFINES(-D_BUILD_GVL_WITH_KINECT_SUPPORT_)
    ICMAKER_ADD_SOURCES(
      Kinect.cpp
    )
    ICMAKER_ADD_HEADERS(
      Kinect.h
    )
    ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
      OpenNI
    )
  ENDIF(OPENNI_FOUND)

ENDIF(PCL_FOUND)

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_HELPERS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
  Boost_FILESYSTEM
)

ICMAKER_DEPENDENCIES( OPTIONAL
  OpenNI
  PCL
)

IF(ROS_FOUND)
  ICMAKER_GLOBAL_CPPDEFINES(-D_BUILD_GVL_WITH_TF_SUPPORT_)
  ICMAKER_ADD_SOURCES(
    tfHelper.cpp
  )
  ICMAKER_ADD_HEADERS(
    tfHelper.h
  )
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
    ROS
  )
ENDIF(ROS_FOUND)


ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/helpers)
# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/helpers/kernels
  kernels/HelperOperations.h
  kernels/MetaPointCloudOperations.h
)
