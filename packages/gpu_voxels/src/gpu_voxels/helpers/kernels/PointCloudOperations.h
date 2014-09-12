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
 * \author  Sebastian Klemm
 * \date    2012-08-23
 *
 * Point Cloud Kernel calls
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_KERNELS_POINT_CLOUD_OPERATIONS_H_INCLUDED
#define GPU_VOXELS_HELPERS_KERNELS_POINT_CLOUD_OPERATIONS_H_INCLUDED
#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>

namespace gpu_voxels {

//! Calculates absolutes[] = base * relatives[]
__global__
void kernelMultiplyMatrixNbyOne(uint32_t nr_of_elements, Matrix4f* base, Matrix4f* relatives, Matrix4f* absolutes);


}
#endif
