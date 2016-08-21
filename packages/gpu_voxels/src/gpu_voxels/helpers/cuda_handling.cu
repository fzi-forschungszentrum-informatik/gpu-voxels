// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-08-31
 *
 *
 */
//----------------------------------------------------------------------
#include "cuda_handling.h"

namespace gpu_voxels {

bool cuCheckForError(const char* file, int line)
{
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
  {
    LOGGING_ERROR(CudaLog,
                  cudaGetErrorString(cuda_error) << "(" << cuda_error << ") in " << file << " on line " << line << "." << endl);
    return false;
  }
  return true;
}


bool cuHandleError(cudaError_t cuda_error, const char* file, int line)
{
  if (cuda_error != cudaSuccess)
  {
    LOGGING_ERROR(CudaLog,
                  cudaGetErrorString(cuda_error) << " in " << file << " on line " << line << "." << endl);
    return false;
  }
  return true;
}

bool cuGetNrOfDevices(int* nr_of_devices)
{
  if (!HANDLE_CUDA_ERROR(cudaGetDeviceCount(nr_of_devices)))
  {
    return false;
  }
  return true;
}

bool cuGetDeviceInfo(cudaDeviceProp* device_properties, int nr_of_devices)
{
  for (int i = 0; i < nr_of_devices; i++)
  {
    if (!HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&device_properties[i] , i)))
    {
      return false;
    }
  }
  return true;
}

bool cuTestAndInitDevice()
{

  // The test requires an architecture SM25 or greater (CDP capable).

  int device_count = 0, device = -1;
  cuGetNrOfDevices(&device_count);
  for (int i = 0; i < device_count; ++i)
  {
    cudaDeviceProp properties;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&properties, i));
    if (properties.major > 2 || (properties.major == 2 && properties.minor >= 0))
    {
      device = i;
      LOGGING_INFO(CudaLog, "Running on GPU " << i << " (" << properties.name << ")" << endl);
      break;
    }
  }
  if (device == -1)
  {
    std::cerr << "No device with SM 2.5 or higher found, which is required for GPU-Voxels.\n"
        << std::endl;
    return false;
  }
  cudaSetDevice(device);
  HANDLE_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
//HANDLE_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  return true;
}

void cuPrintDeviceMemoryInfo()
{
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  //unsigned int free, total, used;
  size_t free, total, used;
  cudaMemGetInfo(&free, &total);
  used = total - free;

  const float byte2mb = (float) 1 / (1024.0 * 1024.0);

  LOGGING_INFO(CudaLog, "Device memory status:" << endl);
  LOGGING_INFO(CudaLog, "-----------------------------------" << endl);
  LOGGING_INFO(CudaLog, "total memory (MB)  : " << (float) total * byte2mb << endl);
  LOGGING_INFO(CudaLog, "free  memory (MB)  : " << (float) free * byte2mb << endl);
  LOGGING_INFO(CudaLog, "used  memory (MB)  : " << (float) used * byte2mb << endl);
  LOGGING_INFO(CudaLog, "-----------------------------------" << endl);
}

} // end of namespace
