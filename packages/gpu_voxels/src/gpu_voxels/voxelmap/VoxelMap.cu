// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-07-10
 *
 */
//----------------------------------------------------------------------/*
#include "VoxelMap.hpp"

namespace gpu_voxels {

template class BitVector<BIT_VECTOR_LENGTH>;
template class BitVoxel<BIT_VECTOR_LENGTH>;

namespace voxelmap {

// ############################### BitVoxelMap ######################################
// Explicit instantiation of template class to link against from other files where this template is used
template class BitVoxelMap<BIT_VECTOR_LENGTH>;
// ##################################################################################

// ############################### TemplateVoxelMap ######################################
// Explicitly instantiate template methods to enable GCC to link agains NVCC compiled objects
template uint32_t TemplateVoxelMap<ProbabilisticVoxel>::collisionCheckWithCounter<ProbabilisticVoxel, DefaultCollider>(
                                                                TemplateVoxelMap<ProbabilisticVoxel>*, DefaultCollider);

// ############################### ProbVoxelMap (inherits from TemplateVoxelMap) ######################################
// Explicitly instantiate template methods to enable GCC to link agains NVCC compiled objects
template void ProbVoxelMap::insertSensorData<BIT_VECTOR_LENGTH>(const PointCloud&, const Vector3f&, const bool, const bool,
                                                                const BitVoxelMeaning, BitVoxel<BIT_VECTOR_LENGTH>*);



// ##################################################################################

} // end of namespace
} // end of namespace
