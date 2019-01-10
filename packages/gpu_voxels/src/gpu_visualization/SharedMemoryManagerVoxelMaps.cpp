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
 * \author  Matthias Wagner
 * \date    2014-07-09
 *
 * \brief This class is for the management of the interprocess communication with the provider.
 */
//----------------------------------------------------------------------
#include <gpu_visualization/SharedMemoryManagerVoxelMaps.h>
#include <gpu_visualization/SharedMemoryManager.h>

using namespace boost::interprocess;
namespace gpu_voxels {
namespace visualization {

SharedMemoryManagerVoxelMaps::SharedMemoryManagerVoxelMaps()
{
  shmm = new SharedMemoryManager(shm_segment_name_voxelmaps, true);
}

SharedMemoryManagerVoxelMaps::~SharedMemoryManagerVoxelMaps()
{
  delete shmm;
}

uint32_t SharedMemoryManagerVoxelMaps::getNumberOfVoxelMapsToDraw()
{
  std::pair<uint32_t*, std::size_t> res = shmm->getMemSegment().find<uint32_t>(
      shm_variable_name_number_of_voxelmaps.c_str());
  if (res.second == 0)
  {
    return 0;
  }

  return *res.first;
}

bool SharedMemoryManagerVoxelMaps::getDevicePointer(void*& dev_pointer, const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string handler_var_name = shm_variable_name_voxelmap_handler_dev_pointer + index_str;
  std::pair<cudaIpcMemHandle_t*, std::size_t> res = shmm->getMemSegment().find<cudaIpcMemHandle_t>(
      handler_var_name.c_str());
  if (res.second == 0)
  {
    return false;
  }
  cudaIpcMemHandle_t handler = *res.first;
  cudaError_t cuda_error = cudaIpcOpenMemHandle((void**) &dev_pointer, (cudaIpcMemHandle_t) handler,
                                                cudaIpcMemLazyEnablePeerAccess);
  // the handle is closed by Visualizer.cu
  return cuda_error == cudaSuccess;
}

bool SharedMemoryManagerVoxelMaps::getVoxelMapDimension(Vector3ui& dim, const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string dimension_var_name = shm_variable_name_voxelmap_dimension + index_str;
  std::pair<Vector3ui*, std::size_t> res = shmm->getMemSegment().find<Vector3ui>(dimension_var_name.c_str());
  if (res.second == 0)
  {
    return false;
  }
  dim = *res.first;
  return true;
}

bool SharedMemoryManagerVoxelMaps::getVoxelMapSideLength(float& voxel_side_length, const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string side_length_var_name = shm_variable_name_voxel_side_length + index_str;
  std::pair<float*, std::size_t> res = shmm->getMemSegment().find<float>(side_length_var_name.c_str());
  if (res.second == 0)
  {
    return false;
  }
  voxel_side_length = *res.first;
  return true;
}

bool SharedMemoryManagerVoxelMaps::getVoxelMapName(std::string& map_name, const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string voxel_map_name_var_name = shm_variable_name_voxelmap_name + index_str;
  std::pair<char*, std::size_t> res_n = shmm->getMemSegment().find<char>(voxel_map_name_var_name.c_str());
  if (res_n.second == 0)
  { /*If the segment couldn't be find or the string is empty*/
    map_name = "voxelmap_" + index_str;
    return false;
  }
  map_name.assign(res_n.first, res_n.second);
  return true;
}

void SharedMemoryManagerVoxelMaps::setVoxelMapDataChangedToFalse(const uint32_t index)
{
  std::string swapped_buffer_name = shm_variable_name_voxelmap_data_changed
      + boost::lexical_cast<std::string>(index);
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

  if (swapped.second)
  {
    *swapped.first = false;
  }
}

bool SharedMemoryManagerVoxelMaps::hasVoxelMapDataChanged(const uint32_t index)
{
  std::string swapped_buffer_name = shm_variable_name_voxelmap_data_changed
      + boost::lexical_cast<std::string>(index);
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

  if (swapped.second != 0)
  {
    return *swapped.first;
  }
  else
  {
    //if the shared variable couldn't be found return true so the voxel map will always be updated.
    return true;
  }
}
bool SharedMemoryManagerVoxelMaps::getVoxelMapType(MapType& type, const uint32_t index)
{
  std::string voxelmap_type_name = shm_variable_name_voxelmap_type + boost::lexical_cast<std::string>(index);
  std::pair<MapType*, std::size_t> res = shmm->getMemSegment().find<MapType>(voxelmap_type_name.c_str());
  if (res.second == 0)
  {
    return false;
  }
  type = *res.first;
  return true;
}

} //end of namespace visualization
} //end of namespace gpu_voxels
