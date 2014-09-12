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

template class BitVector<voxelmap::BIT_VECTOR_LENGTH>;

namespace voxelmap {

// ############################### BitVoxelMap ######################################
// Explicit instantiation of template class to link against from other files where this template is used
template class BitVoxelMap<BIT_VECTOR_LENGTH>;
template class BitVoxel<BIT_VECTOR_LENGTH>;
// ##################################################################################

// ############################### ProbVoxelMap ######################################
// Explicitly instantiate template method
template void ProbVoxelMap::insertSensorData<BIT_VECTOR_LENGTH>(const Vector3f*, const bool, const bool,
                                                                const uint32_t, BitVoxel<BIT_VECTOR_LENGTH>*);
// ##################################################################################

} // end of namespace
} // end of namespace
