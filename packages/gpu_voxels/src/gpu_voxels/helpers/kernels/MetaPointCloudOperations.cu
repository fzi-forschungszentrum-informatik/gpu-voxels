// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------


//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2014-06-17
 *
 * MetaPointCloud kernel calls
 */
//----------------------------------------------------------------------

#include "MetaPointCloudOperations.h"

namespace gpu_voxels {

__global__
void kernelDebugMetaPointCloud(MetaPointCloudStruct* meta_point_clouds_struct)
{

  printf("================== kernelDebugMetaPointCloud DBG ================== \n");


  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0)
  {
    printf("kernelDebugMetaPointCloud DBG: NumClouds: %d \n",
           meta_point_clouds_struct->num_clouds);

    printf("kernelDebugMetaPointCloud DBG: m_dev_ptr_to_clouds_base_addresses: %p \n",
            meta_point_clouds_struct->clouds_base_addresses);

    for(int i = 0; i < meta_point_clouds_struct->num_clouds; i++)
    {
        printf("kernelDebugMetaPointCloud DBG: CloudSize[%d]: %d, clouds_base_addresses[%d]: %p \n",
               i, meta_point_clouds_struct->cloud_sizes[i],
               i, meta_point_clouds_struct->clouds_base_addresses[i]);

        if (meta_point_clouds_struct->cloud_sizes[i] > 0)
        {
          Vector3f min_xyz = meta_point_clouds_struct->clouds_base_addresses[i][0];
          Vector3f max_xyz = meta_point_clouds_struct->clouds_base_addresses[i][0];
          for (uint32_t j = 1; j < meta_point_clouds_struct->cloud_sizes[i]; j++)
          {
            min_xyz.x = min(min_xyz.x, meta_point_clouds_struct->clouds_base_addresses[i][j].x);
            min_xyz.y = min(min_xyz.y, meta_point_clouds_struct->clouds_base_addresses[i][j].y);
            min_xyz.z = min(min_xyz.z, meta_point_clouds_struct->clouds_base_addresses[i][j].z);

            max_xyz.x = max(max_xyz.x, meta_point_clouds_struct->clouds_base_addresses[i][j].x);
            max_xyz.y = max(max_xyz.y, meta_point_clouds_struct->clouds_base_addresses[i][j].y);
            max_xyz.z = max(max_xyz.z, meta_point_clouds_struct->clouds_base_addresses[i][j].z);
          }

          printf("kernelDebugMetaPointCloud DBG: CloudSize[%d] bounds: Min[%f, %f, %f], Max[%f, %f, %f] \n",
                 i, min_xyz.x, min_xyz.y, min_xyz.z, max_xyz.x, max_xyz.y, max_xyz.z);
        }
    }
  }

  printf("================== END kernelDebugMetaPointCloud DBG ================== \n");
}


__global__
void kernelTransformCloud(const Matrix4f *transformation, const Vector3f *startAddress, Vector3f *transformedAddress, uint32_t numberOfPoints)
{
  // copying the transformation matrix to local memory might be faster than accessing it from the global memory
  Matrix4f transform;
  transform = *transformation;

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < numberOfPoints)
  {
    transformedAddress[i] = transform * startAddress[i];
    i += blockDim.x * gridDim.x;
  }
}

__global__
void kernelScaleCloud(const Vector3f scaling, const Vector3f* startAddress, Vector3f* transformedAddress, uint32_t numberOfPoints)
{
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

  while(i < numberOfPoints)
  {
    transformedAddress[i] = scaling * startAddress[i];
    i += blockDim.x * gridDim.x;
  }
}


} // end of namespace gpu_voxels
