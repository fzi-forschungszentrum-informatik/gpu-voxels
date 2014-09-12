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
#ifndef VISNTREE_CPP_
#define VISNTREE_CPP_

#include <gpu_voxels/octree/VisNTree.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <thrust/device_vector.h>

namespace gpu_voxels {
namespace NTree {

thrust::device_vector<Cube> cubes_buffer_1(1);
thrust::device_vector<Cube> cubes_buffer_2(1);

template<typename InnerNode, typename LeafNode>
VisNTree<InnerNode, LeafNode>::VisNTree(MyNTree* ntree, std::string map_name) :
    VisProvider(shm_segment_name_octrees, map_name), m_ntree(ntree), m_shm_memHandle(NULL), m_min_level(
        UINT_MAX), m_shm_superVoxelSize(NULL), m_shm_numCubes(NULL), m_shm_bufferSwapped(NULL)
{

}

template<typename InnerNode, typename LeafNode>
VisNTree<InnerNode, LeafNode>::~VisNTree()
{

}

template<typename InnerNode, typename LeafNode>
bool VisNTree<InnerNode, LeafNode>::visualize(const bool force_repaint)
{
  openOrCreateSegment();
  uint32_t shared_mem_id;
  if (m_shm_memHandle == NULL) // do this only once
  {
    // there should only be one segment of number_of_octrees
    std::pair<uint32_t*, std::size_t> r = m_segment.find<uint32_t>(
        shm_variable_name_number_of_octrees.c_str());
    if (r.second == 0)
    { // if it doesn't exist ..
      m_segment.construct<uint32_t>(shm_variable_name_number_of_octrees.c_str())(1);
      shared_mem_id = 0;
    }
    else
    { // if it exit increase it by one
      shared_mem_id = *r.first;
      (*r.first)++;
    }

    // get shared memory pointer
    std::stringstream id;
    id << shared_mem_id;
    m_shm_superVoxelSize = m_segment.find_or_construct<uint32_t>(shm_variable_name_super_voxel_size.c_str())(
        5);
    m_shm_memHandle = m_segment.find_or_construct<cudaIpcMemHandle_t>(
        std::string(shm_variable_name_octree_handler_dev_pointer + id.str()).c_str())(cudaIpcMemHandle_t());
    m_shm_numCubes = m_segment.find_or_construct<uint32_t>(
        std::string(shm_variable_name_number_cubes + id.str()).c_str())(0);
    m_shm_bufferSwapped = m_segment.find_or_construct<bool>(
        std::string(shm_variable_name_buffer_swapped + id.str()).c_str())(false);
    m_shm_mapName = m_segment.find_or_construct_it<char>(
        std::string(shm_variable_name_octree_name + id.str()).c_str())[m_map_name.size()](m_map_name.data());

  }

  uint32_t tmp = *m_shm_superVoxelSize - 1;
  if (*m_shm_bufferSwapped == false && (tmp != m_min_level || force_repaint))
  {
    m_min_level = tmp;
    Cube* ptr = NULL;
    uint32_t used_size = m_ntree->extractCubes(cubes_buffer_1, NULL, m_min_level);
    ptr = D_PTR(cubes_buffer_1);
    cubes_buffer_1.swap(cubes_buffer_2);

    HANDLE_CUDA_ERROR(cudaIpcGetMemHandle(m_shm_memHandle, ptr));
    *m_shm_numCubes = used_size;
    *m_shm_bufferSwapped = true;

    return true;
  }
  return false;
}

template<typename InnerNode, typename LeafNode>
uint32_t VisNTree<InnerNode, LeafNode>::getResolutionLevel()
{
  if (m_shm_superVoxelSize != NULL)
    return *m_shm_superVoxelSize - 1;
  else
    return 0;
}

}
}

#endif
