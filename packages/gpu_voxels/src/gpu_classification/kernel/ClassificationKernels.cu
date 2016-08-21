#include "ClassificationKernels.h"

#include <stdio.h>

namespace gpu_voxels{
namespace classification {


//__global__
//void kernelKinematicChainTransform(const Matrix4f* transformation_,
//                                   const MetaPointCloudStruct *point_clouds, MetaPointCloudStruct *transformed_point_clouds)
//{
//  // copying the transformation matrix to local memory might be faster than accessing it from the global memory
//  Matrix4f transformation;
//  transformation = *transformation_;

//  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

////  if (i==1) printf("transform (%u): (%f, %f, %f, %f)\n", sizeof(transformation), transformation.a11, transformation.a12, transformation.a13, transformation.a14);

//  // if more than max_nr_of_blocks points are in ptcloud we need a loop
//  while (i < point_clouds->cloud_sizes[joint_to_transform])
//  {

////    applyTransform(transformation, point_clouds[joint_to_transform][i], transformed_point_clouds[joint_to_transform][i]);
//    transformed_point_clouds->clouds_base_addresses[joint_to_transform][i] = transformation * point_clouds->clouds_base_addresses[joint_to_transform][i];
////    if (i==1)
////      printf("transforming (%f, %f, %f) --> (%f, %f, %f)\n",
////             point_clouds->clouds_base_addresses[joint_to_transform][i].x,
////             point_clouds->clouds_base_addresses[joint_to_transform][i].y,
////             point_clouds->clouds_base_addresses[joint_to_transform][i].z,
////             transformed_point_clouds->clouds_base_addresses[joint_to_transform][i].x,
////             transformed_point_clouds->clouds_base_addresses[joint_to_transform][i].y,
////             transformed_point_clouds->clouds_base_addresses[joint_to_transform][i].z);

//    // increment by number of all threads that are running
//    i += blockDim.x * gridDim.x;
//  }
//}

}
}
