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
 * \date    2012-08-31
 *
 *  implementation of cuda_handling.h
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_HANDLING_HPP_INCLUDED
#define GPU_VOXELS_CUDA_HANDLING_HPP_INCLUDED

#include "cuda_handling.h"
#include <stdint.h>

namespace gpu_voxels {

/* Helper functions that can be used for debugging
 * surround with HANDLE_CUDA_ERROR(     ) for error handling
 */

// print single device variable
template<class T>
cudaError_t cuPrintDeviceVariable(T* dev_variable)
{
  T host_variable;
  cudaError_t error = cudaMemcpy(&host_variable, dev_variable, sizeof(T), cudaMemcpyDeviceToHost);
  LOGGING_INFO(Gpu_voxels_helpers, host_variable << endl);
  return error;
}

// as above, with info text
template<class T>
cudaError_t cuPrintDeviceVariable(T* dev_variable, const char text[])
{
  LOGGING_INFO(Gpu_voxels_helpers, text << endl);
  return cuPrintDeviceVariable(dev_variable);
}

// print single device pointer
template<class T>
cudaError_t cuPrintDevicePointer(T* dev_pointer)
{
  T* host_pointer;
  cudaError_t error = cudaMemcpy(&host_pointer, dev_pointer, sizeof(T*), cudaMemcpyDeviceToHost);
  LOGGING_INFO(Gpu_voxels_helpers, host_pointer << endl);
  return error;
}

// as above, with info text
template<class T>
cudaError_t cuPrintDevicePointer(T* dev_pointer, const char text[])
{
  LOGGING_INFO(Gpu_voxels_helpers, text << endl);
  return cuPrintDevicePointer(dev_pointer);
}

// print device array, array_size held on host
template<class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int array_size)
{
  // create host array
  T* host_array = new T[array_size];
  // fill host array with contents of device array
  cudaError_t error = cudaMemcpy(host_array, dev_array, array_size * sizeof(T), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < array_size; i++)
  {
    std::stringstream s;
    s << i << ": " << host_array[i];
    LOGGING_INFO(Gpu_voxels_helpers, s.str() << endl);
  }
  LOGGING_INFO(Gpu_voxels_helpers, endl);
  delete[] host_array;
  return error;
}

template<class T>
void cuAllocAndCopyArray(T* array, uint32_t size, T** dev_pointer)
{
  HANDLE_CUDA_ERROR(cudaMalloc((void** )dev_pointer, sizeof(T) * size));
  cudaDeviceSynchronize();
  HANDLE_CUDA_ERROR(cudaMemcpy(*dev_pointer, &array[0], sizeof(T) * size, cudaMemcpyHostToDevice));
}

// as above, with info text
template<class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int array_size, const char text[])
{
  LOGGING_INFO(Gpu_voxels_helpers, text << endl);
  return cuPrintDeviceArray(dev_array, array_size);
}

// print device array, array size also held on device
template<class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int* device_array_size)
{
  // copy array size
  unsigned int array_size;
  cudaError_t error1 = cudaMemcpy(&array_size, device_array_size, sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost);

  // create host array
  T* host_array = new T[array_size];
  // fill host array with contents of device array
  cudaError_t error2 = cudaMemcpy(host_array, dev_array, array_size * sizeof(T), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < array_size; i++)
  {
    std::stringstream s;
    s << i << ": " << host_array[i];
    LOGGING_INFO(Gpu_voxels_helpers, s.str() << endl);
  }
  LOGGING_INFO(Gpu_voxels_helpers, endl);
  delete[] host_array;
  if (error1 == cudaSuccess)
  {
    return error2;
  }
  else
  {
    return error1;
  }
}

// as above, with info text
template<class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int* device_array_size, const char text[])
{
  LOGGING_INFO(Gpu_voxels_helpers, text << endl);
  return cuPrintDeviceArray(dev_array, device_array_size);
}

} // end of namespace
#endif
