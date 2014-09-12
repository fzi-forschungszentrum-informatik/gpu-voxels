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
 * \author  Florian Drews
 * \date    2014-06-18
 *
 */
//----------------------------------------------------------------------/*
#include <gpu_voxels/vis_interface/VisVoxelMap.h>
#include <gpu_voxels/helpers/cuda_handling.h>
#include <cstdio>

namespace gpu_voxels {

VisVoxelMap::VisVoxelMap(voxelmap::AbstractVoxelMap* voxelmap, std::string map_name) :
    VisProvider(shm_segment_name_voxelmaps, map_name), /**/
    m_voxelmap(voxelmap), /**/
    m_shm_memHandle(NULL), /**/
    m_shm_mapDim(NULL), /**/
    m_shm_VoxelSize(NULL), /**/
    m_shm_voxelmap_type(NULL), /**/
    m_shm_voxelmap_changed(NULL)
{
}

VisVoxelMap::~VisVoxelMap()
{
}

bool VisVoxelMap::visualize(const bool force_repaint)
{
  if (force_repaint)
  {
    openOrCreateSegment();
    uint32_t shared_mem_id;
    if (m_shm_memHandle == NULL)
    {
      // there should only be one segment of number_of_voxelmaps
      std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(
          shm_variable_name_number_of_voxelmaps.c_str());
      if (r.second == 0)
      { // if it doesn't exists ..
        m_segment.construct<uint32_t>(shm_variable_name_number_of_voxelmaps.c_str())(1);
        shared_mem_id = 0;
      }
      else
      { // if it exists increase it by one
        shared_mem_id = *r.first;
        (*r.first)++;
      }
      // get shared memory pointer
      std::stringstream id;
      id << shared_mem_id;
      m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
          std::string(shm_variable_name_voxelmap_handler_dev_pointer + id.str()).c_str())(
          cudaIpcMemHandle_t());
      m_shm_mapDim = m_segment.find_or_construct<Vector3ui>(
          std::string(shm_variable_name_voxelmap_dimension + id.str()).c_str())(Vector3ui(0));
      m_shm_VoxelSize = m_segment.find_or_construct<float>(
          std::string(shm_variable_name_voxel_side_length + id.str()).c_str())(0.0f);
      m_shm_mapName = m_segment.find_or_construct_it<char>(
          std::string(shm_variable_name_voxelmap_name + id.str()).c_str())[m_map_name.size()](
          m_map_name.data());
      m_shm_voxelmap_type = m_segment.find_or_construct<MapType>(
          std::string(shm_variable_name_voxelmap_type + id.str()).c_str())(m_voxelmap->getMapType());

      m_shm_voxelmap_changed = m_segment.find_or_construct<bool>(
          std::string(shm_variable_name_voxelmap_data_changed + id.str()).c_str())(true);

    }
    // first open or create and the set the values
    HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, m_voxelmap->getVoidDeviceDataPtr()));
    *m_shm_mapDim = m_voxelmap->getDimensions();
    *m_shm_VoxelSize = m_voxelmap->getVoxelSideLength();
    *m_shm_voxelmap_changed = true;

//    // wait till data was read by visualizer. Otherwise a
//    while(*m_shm_voxelmap_changed)
//      usleep(10000); // sleep 10 ms

    return true;
  }
  return false;
}

uint32_t VisVoxelMap::getResolutionLevel()
{
  return 0; // todo query correct resolution from visualizer like VisNTree
}

} // end of ns
