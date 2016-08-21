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
 * \date    2016-05-25
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_KERNELS_VOXEL_MAP_TESTS_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_KERNELS_VOXEL_MAP_TESTS_H_INCLUDED

#include <cuda_runtime.h>
#include "VoxelMapTests.h"

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>



namespace gpu_voxels {
namespace voxelmap {
namespace test {

//// Explicitly instantiate template methods to enable GCC to link agains NVCC compiled objects
//void kernelAddressingTest(const ProbabilisticVoxel*, const Vector3ui, const float,
//                          const Vector3f*, const size_t, bool*);

//void kernelAddressingTest(const BitVectorVoxel*, const Vector3ui, const float,
//                          const Vector3f*, const size_t, bool*);

} // end of namespace test
} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
