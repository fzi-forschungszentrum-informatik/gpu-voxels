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
    }
  }
}

} // end of namespace gpu_voxels
