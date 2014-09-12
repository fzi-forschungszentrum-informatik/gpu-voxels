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
 * \author  Florian Drews
 * \date    2014-01-21
 *
 */
//----------------------------------------------------------------------

#ifndef TESTS_H_
#define TESTS_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector_types.h>
#include <vector>
#include <thrust/host_vector.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/DataTypes.h>

namespace gpu_voxels {
namespace NTree {
namespace Test {

static const size_t TEST_RAND_SEED = 942359275UL;

static const uint32_t BRANCHING_FACTOR = 8;
static const uint32_t LEVEL_COUNT = 9;
static const uint32_t NUM_VOXEL = pow(BRANCHING_FACTOR, LEVEL_COUNT - 1);

thrust::host_vector<gpu_voxels::Vector3ui> randomPoints(voxel_count num_points, VoxelID maxValue);

thrust::host_vector<gpu_voxels::Vector3ui> randomCube(
    gpu_voxels::Vector3ui map_dimensions, uint32_t cube_side_length);

void run(std::vector<Vector3f>& points, uint32_t num_points);

void rotate(thrust::host_vector<gpu_voxels::Vector3ui>& points, float angle_degree,
            gpu_voxels::Vector3f translation);

void translate(thrust::host_vector<gpu_voxels::Vector3ui>& points,
               gpu_voxels::Vector3f translation);

}
}
}

#endif /* TESTS_H_ */
