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
 * \brief   Saves all necessary stuff to draw the cube list of an
 * octree or a voxellist.
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_CUBELISTCONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_CUBELISTCONTEXT_H_INCLUDED

#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_visualization/DataContext.h>

namespace gpu_voxels {
namespace visualization {

typedef std::pair<glm::vec4, glm::vec4> colorPair;

class CubelistContext: public DataContext
{
public:

  CubelistContext(std::string map_name)
    : m_d_cubes(NULL), 
      m_number_of_cubes(0)
  {
    m_map_name = map_name;
    m_default_prim = new Cuboid(glm::vec4(0.f, 0.f, 0.f, 1.f),
                                glm::vec3(0.f, 0.f, 0.f),
                                glm::vec3(1.f, 1.f, 1.f));
    m_threads_per_block = dim3(cMAX_THREADS_PER_BLOCK);
    m_num_blocks = dim3(50);
  }

  CubelistContext(Cube* cubes, uint32_t num_cubes, std::string map_name)
  {
    m_map_name = map_name;
    m_d_cubes = cubes;
    m_number_of_cubes = num_cubes;
    m_default_prim = new Cuboid(glm::vec4(0.f, 0.f, 0.f, 1.f),
                                glm::vec3(0.f, 0.f, 0.f),
                                glm::vec3(1.f, 1.f, 1.f));
    m_threads_per_block = dim3(cMAX_THREADS_PER_BLOCK);
    m_num_blocks = dim3(50);
  }

  ~CubelistContext()
  {
  }

  Cube* getCubesDevicePointer()
  {
    return m_d_cubes;
  }

  void setCubesDevicePointer(Cube* cubes)
  {
    m_d_cubes = cubes;
  }

  uint32_t getNumberOfCubes() const
  {
    return m_number_of_cubes;
  }

  void setNumberOfCubes(uint32_t numberOfCubes)
  {
    m_number_of_cubes = numberOfCubes;
  }

  void unmapCubesShm()
  {
    if (m_d_cubes != NULL)
    {
      cudaIpcCloseMemHandle(m_d_cubes);
      m_d_cubes = NULL;
    }
  }

  virtual void updateVBOOffsets()
  {
    thrust::exclusive_scan(m_num_voxels_per_type.begin(), m_num_voxels_per_type.end(), m_vbo_offsets.begin());
    m_d_vbo_offsets = m_vbo_offsets;
  }
  virtual void updateCudaLaunchVariables(Vector3ui supervoxel_size = Vector3ui(1))
  {
    m_threads_per_block = dim3(cMAX_THREADS_PER_BLOCK);
    m_num_blocks = dim3(m_number_of_cubes / cMAX_THREADS_PER_BLOCK + 1);
  }

private:
  // the GPU pointer to the cubes of this context
  Cube* m_d_cubes;
  // the number of cubes in m_cubes
  uint32_t m_number_of_cubes;

};

} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
