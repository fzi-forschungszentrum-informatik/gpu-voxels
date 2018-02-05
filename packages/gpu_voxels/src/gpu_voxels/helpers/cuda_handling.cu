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
#include <gpu_voxels/helpers/common_defines.h>

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


std::string getDeviceInfos()
{
  std::stringstream tmp_stream;

  int nr_of_devices;
  cuGetNrOfDevices(&nr_of_devices);

  if(nr_of_devices > 0)
  {
      std::cout << "Found " << nr_of_devices << " devices." << std::endl;
      cudaDeviceProp* props = new cudaDeviceProp[nr_of_devices];

      cuGetDeviceInfo(props, nr_of_devices);

      for (int i = 0; i < nr_of_devices; i++)
      {
        tmp_stream << "Device Information of GPU " << i << std::endl
                   << "Model: " << props[i].name << std::endl
                   << "Multi Processor Count: " << props[i].multiProcessorCount << std::endl
                   << "Global Memory: " << cBYTE2MBYTE * props[i].totalGlobalMem << " MB" << std::endl
                   << "Total Constant Memory: " << props[i].totalConstMem << std::endl
                   << "Shared Memory per Block: " << props[i].sharedMemPerBlock << " Shared Memory per Multi Processor: " << props[i].sharedMemPerMultiprocessor << std::endl
                   << "Max Threads per Block " << props[i].maxThreadsPerBlock << " Max Threads per Multi Processor: " << props[i].maxThreadsPerMultiProcessor << std::endl
                   << "Registers per Block: " << props[i].regsPerBlock << " Registers per Multi Processor: " <<props[i].regsPerMultiprocessor << std::endl
                   << "Max grid dimensions: [ " << props[i].maxGridSize[0] << ", " << props[i].maxGridSize[1]  << ", " << props[i].maxGridSize[2] << " ]" << std::endl
                   << "Max Block dimension: [ " << props[i].maxThreadsDim[0] << ", " << props[i].maxThreadsDim[1]  << ", " << props[i].maxThreadsDim[2] << " ]" << std::endl
                   << "Warp Size: " << props[i].warpSize << std::endl << std::endl;

        std::cout << "Dev " << i << " = " << tmp_stream.str() << std::endl;
      }

      delete props;
  }
  return tmp_stream.str();
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

std::string getDeviceMemoryInfo()
{
  std::stringstream tmp_stream;
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  //unsigned int free, total, used;
  size_t free, total, used;
  cudaMemGetInfo(&free, &total);
  used = total - free;

  tmp_stream << "Device memory status:" << std::endl;
  tmp_stream << "-----------------------------------" << std::endl;
  tmp_stream << "total memory (MB)  : " << (float) total * cBYTE2MBYTE << std::endl;
  tmp_stream << "free  memory (MB)  : " << (float) free * cBYTE2MBYTE << std::endl;
  tmp_stream << "used  memory (MB)  : " << (float) used * cBYTE2MBYTE << std::endl;
  tmp_stream << "-----------------------------------" << std::endl;

  return tmp_stream.str();
}

} // end of namespace
