// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
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
void kernelTransformSubCloud(uint8_t subcloud_to_transform, const Matrix4f* transformation_,
                          const MetaPointCloudStruct *input_cloud, MetaPointCloudStruct *transformed_cloud)
{
  // copying the transformation matrix to local memory might be faster than accessing it from the global memory
  Matrix4f transformation;
  transformation = *transformation_;

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

//  if (i==1) printf("transform (%u): (%f, %f, %f, %f)\n", sizeof(transformation), transformation.a11, transformation.a12, transformation.a13, transformation.a14);

  // if more than max_nr_of_blocks points are in ptcloud we need a loop
  while (i < input_cloud->cloud_sizes[subcloud_to_transform])
  {

//    applyTransform(transformation, input_cloud[subcloud_to_transform][i], transformed_cloud[subcloud_to_transform][i]);
    transformed_cloud->clouds_base_addresses[subcloud_to_transform][i] = transformation * input_cloud->clouds_base_addresses[subcloud_to_transform][i];
//    if (i==1)
//      printf("transforming (%f, %f, %f) --> (%f, %f, %f)\n",
//             input_cloud->clouds_base_addresses[subcloud_to_transform][i].x,
//             input_cloud->clouds_base_addresses[subcloud_to_transform][i].y,
//             input_cloud->clouds_base_addresses[subcloud_to_transform][i].z,
//             transformed_cloud->clouds_base_addresses[subcloud_to_transform][i].x,
//             transformed_cloud->clouds_base_addresses[subcloud_to_transform][i].y,
//             transformed_cloud->clouds_base_addresses[subcloud_to_transform][i].z);

    // increment by number of all threads that are running
    i += blockDim.x * gridDim.x;
  }
}

__global__
void kernelTransformCloud(const Matrix4f* transformation_, const MetaPointCloudStruct *input_cloud, MetaPointCloudStruct *transformed_cloud)
{
  // copying the transformation matrix to local memory might be faster than accessing it from the global memory
  Matrix4f transformation;
  transformation = *transformation_;

  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

//  if (i==1) printf("transform (%u): (%f, %f, %f, %f)\n", sizeof(transformation), transformation.a11, transformation.a12, transformation.a13, transformation.a14);

  // if more than max_nr_of_blocks points are in ptcloud we need a loop
  while (i < input_cloud->accumulated_cloud_size)
  {
    transformed_cloud->clouds_base_addresses[0][i] = transformation * input_cloud->clouds_base_addresses[0][i];

    // increment by number of all threads that are running
    i += blockDim.x * gridDim.x;
  }
}


} // end of namespace gpu_voxels
