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
  cuda_datatypes.h
  common_defines.h
  cuda_handling.h
  CudaMath.h
  cuda_handling.hpp
  PointcloudFileHandler.h
  BinvoxFileReader.h
  FileReaderInterface.h
  PcdFileReader.h
  XyzFileReader.h
  MetaPointCloud.h
  BitVector.h
  BitVector.hpp
  CompileIssues.h
  )

ICMAKER_ADD_SOURCES(
  GlobalLock.cpp
  cuda_datatypes.cpp
  PointcloudFileHandler.cpp
  BinvoxFileReader.cpp
  PcdFileReader.cpp
  XyzFileReader.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  CompileIssues.h
  cuda_handling.h
  cuda_handling.hpp
  cuda_handling.cu
  CudaMath.h
  CudaMath.cu
  kernels/PointCloudOperations.h
  kernels/PointCloudOperations.cu
  MetaPointCloud.cu
  kernels/MetaPointCloudOperations.h
  kernels/MetaPointCloudOperations.cu
  BitVector.h
  BitVector.hpp
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGPU_VOXELS_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
  PCL
  )

ICMAKER_DEPENDENCIES( OPTIONAL
  OpenNI
  )

IF(OpenNI_FOUND)
  ICMAKER_GLOBAL_CPPDEFINES(-D_BUILD_GVL_WITH_KINECT_SUPPORT_)
  ICMAKER_ADD_SOURCES(
    Kinect.cpp
  )
  ICMAKER_ADD_HEADERS(
    Kinect.h
  )
ENDIF(OpenNI_FOUND)


ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels/helpers)
# Include files here that are needed when the package is installed.
ICMAKER_INSTALL_HEADER_EXTRAS(gpu_voxels/helpers/kernels
  kernels/PointCloudOperations.h
  kernels/MetaPointCloudOperations.h
)
