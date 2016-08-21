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
 * \date    2016-06-05
 *
 * General Kernel calls
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_KERNELS_HELPER_OPERATIONS_H_INCLUDED
#define GPU_VOXELS_HELPERS_KERNELS_HELPER_OPERATIONS_H_INCLUDED
#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>


namespace gpu_voxels {

__global__
void kernelCompareMem(const void* lhs, const void* rhs, uint32_t size_in_byte, bool *results);

}
#endif
