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
 *   Do not include this file. 
 *   Use cuda_handing.hpp instead
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_CUDA_HANDLING_H_INCLUDED
#define GPU_VOXELS_CUDA_HANDLING_H_INCLUDED

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>
#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>
#include <gpu_voxels/logging/logging_cuda.h>


namespace gpu_voxels {

/* Implementation of frequently needed CUDA functions */

#define CHECK_CUDA_ERROR()        cuCheckForError(__FILE__, __LINE__)
//! Shortcut to check for an active cuda error
bool cuCheckForError(const char* file, int line);

#define HANDLE_CUDA_ERROR(error)        cuHandleError(error, __FILE__, __LINE__)
//! Shortcut useful for error handling
bool cuHandleError(cudaError_t cuda_error, const char* file, int line);

//! Returns the number of available CUDA devices
bool cuGetNrOfDevices(int* nr_of_devices);

//! Gives information on available CUDA devices
bool cuGetDeviceInfo(cudaDeviceProp* device_properties, int nr_of_devices);

//! Returns a string with technical information about all devices
std::string getDeviceInfos();

//! Checks if Compute Capability 2.5 or greater is available. WARNING: This sets caching behaviour!
bool cuTestAndInitDevice();

//! Returns a string with Memory info and usage
std::string getDeviceMemoryInfo();

/* Helper functions that can be used for debugging
   surround all functions with HANDLE_CUDA_ERROR(     ) */

//! print single device variable
template <class T>
cudaError_t cuPrintDeviceVariable(T* dev_variable);

//! print single device variable, with info text
template <class T>
cudaError_t cuPrintDeviceVariable(T* dev_variable, const char text[]);

//! print single device pointer
template <class T>
cudaError_t cuPrintDevicePointer(T* dev_pointer);

//! print single device pointer, with info text
template <class T>
cudaError_t cuPrintDevicePointer(T* dev_pointer, const char text[]);

//! print device array, array_size held on host
template <class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int array_size);
//! print device array, array_size held on host, with info text
template <class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int array_size, const char text[]);

//! prints device array, array size also held on device
template <class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int* device_array_size);

//! prints device array, array size also held on device, with info text
template <class T>
cudaError_t cuPrintDeviceArray(T* dev_array, unsigned int* device_array_size, const char text[]);


} // end of namespace
#endif

