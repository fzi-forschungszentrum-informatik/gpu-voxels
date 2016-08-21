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
 * \date    2014-02-10
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_DATACONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_DATACONTEXT_H_INCLUDED

#include <vector_types.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <glm/glm.hpp>

#include <gpu_visualization/Primitive.h>

namespace gpu_voxels {
namespace visualization {

typedef std::pair<glm::vec4, glm::vec4> colorPair;
class DataContext
{

public:

  DataContext() :
      m_map_name(""),m_draw_context(true), m_vbo(), m_vbo_draw_able(false), m_cur_vbo_size(1), m_max_vbo_size(0),
      m_cuda_ressources(), m_occupancy_threshold(0), m_translation_offset(0.f), m_total_num_voxels(0), m_num_voxels_per_type()
  {

    m_threads_per_block = dim3(10, 10, 10);
    m_num_blocks = dim3(1, 1, 1);

    m_default_prim = NULL;

    //insert some default colors
    // The ordering has to be the same as for BitVoxelMeaning enum
    // defined in gpu_voxels/helpers/common_defines.h
    colorPair p;
    p.first = p.second = glm::vec4(1.f, 1.f, 0.f, 1.f);
    m_colors.push_back(p);/*color for voxel type eBVM_FREE yellow*/

    p.first = p.second = glm::vec4(0.f, 1.f, 0.f, 1.f);
    m_colors.push_back(p);/*color for voxel type eBVM_OCCUPIED green*/

    p.first = p.second = glm::vec4(1.f, 0.f, 0.f, 1.f);
    m_colors.push_back(p);/*color for voxel type eBVM_COLLISION red*/

    p.first = p.second = glm::vec4(1.f, 0.f, 1.f, 1.f);
    m_colors.push_back(p);/*color for voxel type eBVM_UNKNOWN magenta*/


    // swept volume colors blend altering fashion from yellow to blue,
    // and from
    float increment = 1.0 / float(eBVM_SWEPT_VOLUME_END - eBVM_SWEPT_VOLUME_START);
    float change = 0.0;
    size_t step = 0;
    for(size_t i = eBVM_SWEPT_VOLUME_START; i <= eBVM_SWEPT_VOLUME_END; ++i)
    {
      change = step * increment;
      if(step%2)
      {
        p.first = p.second = glm::vec4(1.f - change, 1.f, 0.f + change, 1.f); // yellow to light green
      }else{
        p.first = p.second = glm::vec4(1.f - change, 0.f, 0.f + change, 1.f); // red to blue
      }
      ++step;
      m_colors.push_back(p);
    }

    p.first = p.second = glm::vec4(0.f, 1.f, 1.f, 1.f);
    m_colors.push_back(p);/*color for voxel type eBVM_UNDEFINED(255) cyan*/

    m_num_voxels_per_type.resize(MAX_DRAW_TYPES);
    m_d_num_voxels_per_type = m_num_voxels_per_type;

    m_vbo_segment_voxel_capacities.resize(MAX_DRAW_TYPES);
    m_d_vbo_segment_voxel_capacities = m_vbo_segment_voxel_capacities;

    m_vbo_offsets.resize(MAX_DRAW_TYPES);
    m_d_vbo_offsets = m_vbo_offsets;

    m_types_segment_mapping = thrust::host_vector<uint8_t>(MAX_DRAW_TYPES, 0);
    m_has_draw_type_flipped = true;
  }

  virtual ~DataContext()
  {
    delete m_default_prim;
  }

  void updateTotalNumVoxels()
  {
    uint32_t res = 0;
    for (size_t i = 0; i < m_num_voxels_per_type.size(); ++i)
    {
      res += m_num_voxels_per_type[i];
    }
    m_total_num_voxels = res;
  }

  virtual void updateVBOOffsets()
  {
  }

  virtual void updateCudaLaunchVariables(Vector3ui supervoxel_size = Vector3ui(1))
  {
  }

  /**
   * Returns the number of vertices of the specified type in the current view.
   * If i is not a valid type 0 will be returned.
   */
  inline uint32_t getNumberOfVerteciesOfType(uint32_t i)
  {
    return i < m_num_voxels_per_type.size() ? m_num_voxels_per_type[i] * 36 : 0;
  }

  /**
   * Returns the number of vertices in the current view.
   */
  inline uint32_t getNumberOfVertecies()
  {
    return m_total_num_voxels * 36;
  }

  /**
   * Returns the size of all vertices in the current view in byte.
   */
  inline uint32_t getSizeOfVertecies()
  {
    return m_total_num_voxels * 36/*vertices per voxel*/* 3 * sizeof(float)/*size of a vertex*/;
  }

  inline size_t getSizeForBuffer()
  {
    return m_total_num_voxels * SIZE_OF_TRANSLATION_VECTOR;
  }

  uint32_t getOffset(uint32_t i)
  {
    return i < m_vbo_offsets.size() ? m_vbo_offsets[i] : 0;
  }

  // the name of the data structure
  std::string m_map_name;

  //determines if the data context should be drawn
  bool m_draw_context;

  // contains the colors for each type
  thrust::host_vector<colorPair> m_colors;
  // the OpenGL buffer for this data structure
  GLuint m_vbo;
  // indicates if the vbo may be drawn right now
  bool m_vbo_draw_able;
  // the current size of the VBO
  size_t m_cur_vbo_size;
  // the maximum size of the vbo <=> 0 is no limit
  size_t m_max_vbo_size;
  // the cudaGraphicsResource for this data structure
  cudaGraphicsResource* m_cuda_ressources;
  // the default primitive of this context
  Primitive* m_default_prim;

  // the minimum occupancy probability for the context
  Probability m_occupancy_threshold;

  // an offset for the data structure
  glm::vec3 m_translation_offset;

  // total number of occupied voxels in the current view <=> sum(num_voxels_per_type)
  uint32_t m_total_num_voxels;
  //number of occupied voxels of each type
  thrust::host_vector<uint32_t> m_num_voxels_per_type;
  thrust::device_vector<uint32_t> m_d_num_voxels_per_type;

  //the vbo segment sizes
  thrust::host_vector<uint32_t> m_vbo_segment_voxel_capacities;
  thrust::device_vector<uint32_t> m_d_vbo_segment_voxel_capacities;

  thrust::host_vector<uint32_t> m_vbo_offsets;
  thrust::device_vector<uint32_t> m_d_vbo_offsets;

  // mapping from type to segment
  thrust::host_vector<uint8_t> m_types_segment_mapping;
  bool m_has_draw_type_flipped;

  //cuda kernel launch variable
  dim3 m_threads_per_block;
  dim3 m_num_blocks;
};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
