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
#ifndef GPU_VOXELS_VOXELMAP_KERNELS_VOXEL_MAP_TESTS_HPP_INCLUDED
#define GPU_VOXELS_VOXELMAP_KERNELS_VOXEL_MAP_TESTS_HPP_INCLUDED

#include "VoxelMapTests.h"
#include "VoxelMapOperations.h"
#include <gpu_voxels/voxel/BitVoxel.hpp>

namespace gpu_voxels {
namespace voxelmap {
namespace test {


template<class Voxel>
__global__
void kernelAddressingTest(const Voxel* voxelmap_base_address, const Vector3ui dimensions, const float voxel_side_length,
                          const Vector3f *testpoints, const size_t testpoints_size, bool* success)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < testpoints_size; i += gridDim.x * blockDim.x)
  {
    Vector3ui test_ccords = mapToVoxels(voxel_side_length, testpoints[i]);
    Voxel* testvoxel = getVoxelPtr(voxelmap_base_address, dimensions, test_ccords.x, test_ccords.y, test_ccords.z);
    Vector3ui int_coords = mapToVoxels(voxelmap_base_address, dimensions, testvoxel);
    Vector3f center = getVoxelCenter(voxel_side_length, int_coords);

//    printf("TestCoord    (%f,%f,%f)\n",testpoints[i].x, testpoints[i].y, testpoints[i].z);
//    printf("TestIntCoord (%d,%d,%d)\n",int_coords.x, int_coords.y, int_coords.z);
//    printf("ReturnCoord  (%f,%f,%f)\n",center.x, center.y, center.z);

    if ((abs(center.x - testpoints[i].x) > voxel_side_length / 2.0) ||
        (abs(center.y - testpoints[i].y) > voxel_side_length / 2.0) ||
        (abs(center.z - testpoints[i].z) > voxel_side_length / 2.0))
    {
      *success = false;
    }
  }
}

} // end of namespace test
} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
