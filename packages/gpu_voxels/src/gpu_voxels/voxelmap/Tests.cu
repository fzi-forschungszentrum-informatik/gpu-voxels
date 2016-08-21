// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann
 * \date    2016-05-25
 *
 */
//----------------------------------------------------------------------/*
#include "Tests.hpp"

namespace gpu_voxels {

template class BitVoxel<BIT_VECTOR_LENGTH>;

namespace voxelmap {
namespace test {

// ##################################################################################

// Explicitly instantiate template methods to enable GCC to link agains NVCC compiled objects
template void triggerAddressingTest<ProbabilisticVoxel>(Vector3ui, float, size_t, bool*);
template void triggerAddressingTest<BitVectorVoxel>(Vector3ui, float, size_t, bool*);

// ##################################################################################

} // end of namespace
} // end of namespace
} // end of namespace
