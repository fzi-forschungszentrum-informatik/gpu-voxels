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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-07
*
*/
//----------------------------------------------------------------------

#include "SharedMemoryManagerVoxelLists.h"

#include <gpu_visualization/SharedMemoryManager.h>
#include <gpu_visualization/logging/logging_visualization.h>

#include <boost/lexical_cast.hpp>

using namespace boost::interprocess;
namespace gpu_voxels {
namespace visualization {

SharedMemoryManagerVoxelLists::SharedMemoryManagerVoxelLists()
{
  shmm = new SharedMemoryManager(shm_segment_name_voxellists, true);
}

SharedMemoryManagerVoxelLists::~SharedMemoryManagerVoxelLists()
{
  delete shmm;
}

uint32_t gpu_voxels::visualization::SharedMemoryManagerVoxelLists::getNumberOfVoxelListsToDraw()
{
  std::pair<uint32_t*, std::size_t> res = shmm->getMemSegment().find<uint32_t>(
      shm_variable_name_number_of_voxellists.c_str());
  if (res.second == 0)
  {
    return 0;
  }

  return *res.first;
}

bool SharedMemoryManagerVoxelLists::getVoxelListName(std::string& list_name, const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string voxel_list_name_var_name = shm_variable_name_voxellist_name + index_str;
  std::pair<char*, std::size_t> res_n = shmm->getMemSegment().find<char>(voxel_list_name_var_name.c_str());
  if (res_n.second == 0)
  { /*If the segment couldn't be find or the string is empty*/
    list_name = "voxellist_" + index_str;
    return false;
  }
  list_name.assign(res_n.first, res_n.second);
  return true;
}

bool SharedMemoryManagerVoxelLists::getVisualizationData(Cube*& cubes, uint32_t& size, const uint32_t index)
{
  bool error = false;

  std::string handler_name = shm_variable_name_voxellist_handler_dev_pointer + boost::lexical_cast<std::string>(index);
  std::string number_cubes_name = shm_variable_name_voxellist_num_voxels + boost::lexical_cast<std::string>(index);

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

void SharedMemoryManagerVoxelLists::setBufferSwappedToFalse(const uint32_t index)
{
  std::string swapped_buffer_name = shm_variable_name_voxellist_buffer_swapped
      + boost::lexical_cast<std::string>(index);
  std::pair<bool*, std::size_t> swapped = shmm->getMemSegment().find<bool>(swapped_buffer_name.c_str());
  if (swapped.second)
  {
    *swapped.first = false;
  }
}

bool SharedMemoryManagerVoxelLists::hasBufferSwapped(const uint32_t index)
{
  std::string index_str = boost::lexical_cast<std::string>(index);
  std::string swapped_buffer_name = shm_variable_name_voxellist_buffer_swapped + index_str;
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


} //end of namespace visualization
} //end of namespace gpu_voxels
