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
 *
 */
//----------------------------------------------------------------------
#include <gpu_visualization/SharedMemoryManagerOctrees.h>
#include <gpu_visualization/SharedMemoryManager.h>

using namespace boost::interprocess;
namespace gpu_voxels {
namespace visualization {

SharedMemoryManagerOctrees::SharedMemoryManagerOctrees()
{
  shmm = new SharedMemoryManager(shm_segment_name_octrees, true);
}

SharedMemoryManagerOctrees::~SharedMemoryManagerOctrees()
{
  delete shmm;
}

std::string SharedMemoryManagerOctrees::getNameOfOctree(const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string octree_name_var_name = shm_variable_name_octree_name + index_str;
  // Find the name of the voxel map
  std::pair<char*, std::size_t> res_n = shmm->getMemSegment().find<char>(octree_name_var_name.c_str());
  std::string map_name;
  if (res_n.second == 0)
  { /*If the segment couldn't be find or the string is empty*/
    map_name = "octree_" + index_str;
  }
  else
  {
    map_name.assign(res_n.first, res_n.second);
  }
  return map_name;
}

uint32_t SharedMemoryManagerOctrees::getNumberOfOctreesToDraw()
{
  std::pair<uint32_t*, std::size_t> res = shmm->getMemSegment().find<uint32_t>(
      shm_variable_name_number_of_octrees.c_str());
  if (res.second == 0)
  {
    return 0;
  }
  return *res.first;
}

bool SharedMemoryManagerOctrees::getOctreeVisualizationData(Cube*& cubes, uint32_t& size, uint32_t index)
{
  bool error = false;

  std::string handler_name = shm_variable_name_octree_handler_dev_pointer + boost::lexical_cast<std::string>(index);
  std::string number_cubes_name = shm_variable_name_number_cubes + boost::lexical_cast<std::string>(index);

  //Find the handler object
  std::pair<cudaIpcMemHandle_t*, std::size_t> res_h = shmm->getMemSegment().find<cudaIpcMemHandle_t>(
      handler_name.c_str());
  error = res_h.second == 0;
  Cube* dev_data_pointer;

  if (!error)
  {
    cudaIpcMemHandle_t handler = *res_h.first;
    // get to device data pointer from the handler
    cudaError_t cuda_error = cudaIpcOpenMemHandle((void**) &dev_data_pointer, (cudaIpcMemHandle_t) handler,
                                                  cudaIpcMemLazyEnablePeerAccess);
    // the handle is closed by Visualizer.cu
    if (cuda_error == cudaSuccess)
    {
      //Find the number of cubes
      std::pair<uint32_t*, std::size_t> res_d = shmm->getMemSegment().find<uint32_t>(number_cubes_name.c_str());
      error = res_d.second == 0;
      if (!error)
      {
        cubes = dev_data_pointer;
        size = *res_d.first;
        return true;
      }
    }
  }
  /*If an error occurred */
  cudaIpcCloseMemHandle(dev_data_pointer);
  return false;
}

void SharedMemoryManagerOctrees::setOctreeBufferSwappedToFalse(const uint32_t index)
{
  std::string swapped_buffer_name = shm_variable_name_octree_buffer_swapped
      + boost::lexical_cast<std::string>(index);
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());
  if (swapped.second)
  {
    *swapped.first = false;
  }
}

bool SharedMemoryManagerOctrees::hasOctreeBufferSwapped(const uint32_t index)
{
  std::string swapped_buffer_name = shm_variable_name_octree_buffer_swapped
      + boost::lexical_cast<std::string>(index);
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());

  if (swapped.second != 0)
  {
    return *swapped.first;
  }
  else
  {
    //std::cout << "Error while finding the shared swapped buffer variable." << std::endl;
    return false;
  }
}

void SharedMemoryManagerOctrees::setOctreeOccupancyThreshold(const uint32_t index, Probability threshold)
{
  std::string threshold_buffer_name = shm_variable_name_occupancy_threshold
      + boost::lexical_cast<std::string>(index);
  shmm->getMemSegment().find_or_construct<Probability>(threshold_buffer_name.c_str())(threshold);
}

bool SharedMemoryManagerOctrees::getSuperVoxelSize(uint32_t & sdim)
{
  std::pair<uint32_t*, std::size_t> res_s = shmm->getMemSegment().find<uint32_t>(shm_variable_name_super_voxel_size.c_str());
  if (res_s.second != 0)
  {
    sdim = std::pow(2,*(res_s.first));
    return true;
  }
  return false;

}

void SharedMemoryManagerOctrees::setSuperVoxelSize(uint32_t sdim)
{
  std::pair<uint32_t*, std::size_t> res_s = shmm->getMemSegment().find<uint32_t>(shm_variable_name_super_voxel_size.c_str());
  if (res_s.second != 0)
  {
    *(res_s.first) = (uint32_t) (std::log(sdim) / std::log(2)) + 1;
  }
}

void SharedMemoryManagerOctrees::setView(Vector3ui start_voxel, Vector3ui end_voxel)
{
  shmm->getMemSegment().find_or_construct<Vector3ui>(shm_variable_name_view_start_voxel.c_str())(start_voxel);
  shmm->getMemSegment().find_or_construct<Vector3ui>(shm_variable_name_view_end_voxel.c_str())(end_voxel);
}

} //end of namespace visualization
} //end of namespace gpu_voxels
