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
 * \author  Matthias Wagner
 * \date    2014-12-17
 *
 * \brief This file contains the CUDA-Kernels of the visualizer.
 *
 */
//----------------------------------------------------------------------/*

#ifndef GPU_VOXELS_VISUALIZATION_KERNELS_VOXEL_MAP_VISUALIZER_OPERATIONS_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_KERNELS_VOXEL_MAP_VISUALIZER_OPERATIONS_H_INCLUDED

#include <cuda_runtime.h>
#include <assert.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
//#include <gpu_voxels/voxelmap/Voxel.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_visualization/visualizerDefines.h>

namespace gpu_voxels {
namespace visualization {

//////////////////////////////////// CUDA device functions /////////////////////////////////////////
__host__
__device__
inline Vector3ui indexToXYZ(const uint32_t index, const Vector3ui dim)
{
  Vector3ui r;
  r.z = index / (dim.x * dim.y);
  r.y = index / dim.x - dim.y * r.z;
  r.x = index - dim.x * dim.y * r.z - dim.x * r.y;
  return r;
}
//////////////////////////////////// CUDA kernel functions /////////////////////////////////////////

__global__ void fill_vbo_without_precounting(voxelmap::ProbabilisticVoxel* voxelMap, Vector3ui dim_voxel_map,
                                             Vector3ui dim_super_voxel, Vector3ui start_voxel,
                                             Vector3ui end_voxel, uint8_t occupancy_threshold, float4* vbo,
                                             uint32_t* vbo_offsets, uint32_t* vbo_limits,
                                             uint32_t* write_index, uint8_t*, uint8_t* prefixes);

__global__ void fill_vbo_without_precounting(voxelmap::BitVectorVoxel* voxelMap, Vector3ui dim_voxel_map,
                                             Vector3ui dim_super_voxel, Vector3ui start_voxel,
                                             Vector3ui end_voxel, uint8_t occupancy_threshold, float4* vbo,
                                             uint32_t* vbo_offsets, uint32_t* vbo_limits,
                                             uint32_t* write_index, uint8_t*, uint8_t* prefixes);

__global__ void fill_vbo_with_octree(Cube* cubes, uint32_t size, float4* vbo, uint32_t* vbo_offsets,
                                     uint32_t* write_index, uint8_t* draw_voxel_type, uint8_t* prefixes);

__global__ void calculate_cubes_per_type(Cube* cubes, uint32_t size, uint32_t* cubes_per_type,
                                         uint8_t* draw_voxel_type, uint8_t* prefixes);
} // end of ns
} // end of ns

#endif
