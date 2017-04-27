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
#ifndef GPU_VOXELS_HELPERS_KERNELS_METAPOINTCLOUDOPERATIONS_H_INCLUDED
#define GPU_VOXELS_HELPERS_KERNELS_METAPOINTCLOUDOPERATIONS_H_INCLUDED
#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>

namespace gpu_voxels {

__global__
void kernelDebugMetaPointCloud(MetaPointCloudStruct* meta_point_clouds_struct);

/*!
 * \brief kernelTransformCloud transforms numberOfPoints Points starting at startAddress
 * \param transformation The transformation to be applied
 * \param startAddress address of the points to be transformed
 * \param transformedAddress address where to store the transformed points. Can be the same as the input_cloud
 * \param numberOfPoints number of points to be transformed
*/
__global__
void kernelTransformCloud(const Matrix4f* transformation, const Vector3f* startAddress, Vector3f* transformedAddress, uint32_t numberOfPoints);

/*!
 * \brief kernelScaleCloud scaled numberOfPoints Points starting at startAddress
 * \param scaling The scaling factors to be applied
 * \param startAddress address of the points to be transformed
 * \param transformedAddress address where to store the transformed points. Can be the same as the input_cloud
 * \param numberOfPoints number of points to be transformed
*/
__global__
void kernelScaleCloud(const Vector3f scaling, const Vector3f* startAddress, Vector3f* transformedAddress, uint32_t numberOfPoints);


}
#endif
