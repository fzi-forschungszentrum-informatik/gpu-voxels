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
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

namespace gpu_voxels {
namespace voxelmap {
namespace test {

////! Function that tests 3d -> 1d mapping of voxel map storage
template<class Voxel>
__global__
void kernelAddressingTest(const Voxel* voxelmap_base_address, const Vector3ui dimensions, const float voxel_side_length,
                          const Vector3f *testpoints, const size_t testpoints_size, bool* success);


} // end of namespace test
} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
