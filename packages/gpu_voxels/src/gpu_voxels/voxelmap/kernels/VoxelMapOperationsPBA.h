// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
/*
===============================================================================

Copyright (c) 2010, School of Computing, National University of Singapore.
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the National University of University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \brief   Original PBA implementation by Cao Thanh Tung. Modified by others.
 * \author  Cao Thanh Tung
 * \author  Christian Jülg
 * \date    2010-2016
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_PBA_H_INCLUDED
#define ICL_PLANNING_GPU_KERNELS_VOXELMAP_OPERATIONS_PBA_H_INCLUDED

namespace gpu_voxels {
namespace voxelmap {

/**
 * PBA phase 1: flood obstacles within band slice to right
 */
template<typename InputIterator1, typename InputIterator2>
__global__ void kernelPBAphase1FloodZ(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int band_size);

/**
 * PBA phase 1: collect possible begins and ends to this band from other bands
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase1PropagateInterband(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int bandSize);

/**
 * PBA phase 1: update band using new top and bottom pixels; top and bottom in input; transform output voxel
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase1Update(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int bandSize);

/**
 * PBA phase 2
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase2ProximateBackpointers(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int bandSize);

template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase2CreateForwardPointers(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int bandSize);

template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase2MergeBands(InputIterator1 input, InputIterator2 output, const Vector3ui dims, int bandSize);

/**
 * PBA phase 3
 */
template<typename InputIterator1, typename InputIterator2>
__global__
void kernelPBAphase3Distances(InputIterator1 input, InputIterator2 output, const Vector3ui dims, bool calc_distance);

/**
 * in-place transpose x/y coordinates within every z-layer
 */
template<typename InputIterator>
__global__
void kernelPBA3DTransposeXY(InputIterator input);

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
