// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
*
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-03-03
*
*/
//----------------------------------------------------------------------

#ifndef SHARED_MEM_CUH
#define SHARED_MEM_CUH

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/voxel/AbstractVoxel.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>

// non-specialized class template
template <typename T>
struct SharedVoxel
{
public:
  //! @brief Return a pointer to the runtime-sized shared memory array.
  //! @returns Pointer to runtime-sized shared memory array
  __device__ T* getPointer()
  {
    extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
    Error_UnsupportedType();
    return (T*)0;
  }
};

// specialization for AbstractVoxel
template <>
struct SharedVoxel <gpu_voxels::AbstractVoxel>
{
public:
  __device__ gpu_voxels::AbstractVoxel* getPointer()
  {
    extern __shared__ gpu_voxels::AbstractVoxel abstract_mem[];
    return abstract_mem;
  }
};


// specialization for BitVoxel
template <>
struct SharedVoxel <gpu_voxels::BitVoxel<gpu_voxels::BIT_VECTOR_LENGTH> >
{
public:
  __device__ gpu_voxels::BitVoxel<gpu_voxels::BIT_VECTOR_LENGTH>* getPointer()
  {
    extern __shared__ gpu_voxels::BitVoxel<gpu_voxels::BIT_VECTOR_LENGTH> bit_mem[];
    return bit_mem;
  }
};

// specialization for ProbabilisticVoxel
template <>
struct SharedVoxel <gpu_voxels::ProbabilisticVoxel>
{
public:
  __device__ gpu_voxels::ProbabilisticVoxel* getPointer()
  {
    extern __shared__ gpu_voxels::ProbabilisticVoxel probabilistic_mem[];
    return probabilistic_mem;
  }
};





#endif // SHARED_VOXEL_CUH
