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
 * \date    2014-01-20
 *
 * \brief  Contains the parameter for the visualizer.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_VISUALIZERCONTEXT_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_VISUALIZERCONTEXT_H_INCLUDED

#include <vector_types.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <gpu_visualization/Camera.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>

#include <gpu_visualization/VoxelmapContext.h>
#include <gpu_visualization/CubelistContext.h>
#include <gpu_visualization/PrimitiveArrayContext.h>

namespace gpu_voxels {
namespace visualization {

struct VisualizerContext
{
  VisualizerContext()
    : m_dim_svoxel(1, 1, 1),
      m_min_view_dim(25.f),
      m_dim_view(m_min_view_dim),
      m_max_voxelmap_dim(0),
      m_view_start_voxel_pos(0, 0, 0),
      m_view_end_voxel_pos(10, 10, 10),
      m_max_xyz_to_draw(m_view_end_voxel_pos),
      m_min_xyz_to_draw(m_view_start_voxel_pos),
      m_interpolation_length(25.f),
      m_grid_vbo(0),
      m_focus_point_vbo(0),
      m_grid_distance(10.f),
      m_grid_height(-0.2f),
      m_grid_max_x(1000.f),
      m_grid_max_y(1000.f),
      m_draw_edges_of_triangels(false),
      m_draw_filled_triangles(true),
      m_draw_whole_map(false),
      m_draw_grid(true),
      m_draw_collison_depth_test_always(false),
      m_getCamTargetFromShrMem(false),
      m_camera(NULL),
      m_background_color(0.0f, 0.0f, 0.f, 1.f),
      m_edge_color(0.f, 0.f, 1.f, 1.f),
      m_grid_color(0.f, 0.f, 0.5f, 1.f),
      m_lighting(true),
      m_light_intensity(2500.f),
      m_slice_axis(0),
      m_slice_axis_position(0),
      m_distance_drawmode(0)
  {
    m_draw_types = thrust::host_vector<uint8_t>(MAX_DRAW_TYPES, 0);
    m_draw_types[eBVM_OCCUPIED] = (uint8_t) 1;
    m_draw_types[eBVM_COLLISION] = (uint8_t) 1;

    thrust::fill(m_draw_types.begin()+static_cast<uint>(eBVM_SWEPT_VOLUME_START),
                 m_draw_types.begin()+static_cast<uint>(eBVM_UNDEFINED), 1); // this does not include eBVM_UNDEFINED but only eBVM_SWEPT_VOLUME_END!!
  }

  ~VisualizerContext()
  {
    for (std::vector<VoxelmapContext*>::iterator it = m_voxel_maps.begin(); it != m_voxel_maps.end(); ++it)
    {
      delete *it;
    }
    for (std::vector<CubelistContext*>::iterator it = m_voxel_lists.begin(); it != m_voxel_lists.end(); ++it)
    {
      delete *it;
    }
    for (std::vector<CubelistContext*>::iterator it = m_octrees.begin(); it != m_octrees.end(); ++it)
    {
      delete *it;
    }
    for (std::vector<PrimitiveArrayContext*>::iterator it = m_prim_arrays.begin(); it != m_prim_arrays.end(); ++it)
    {
      delete *it;
    }
    delete m_camera;
  }

  // the voxel maps of this context
  std::vector<VoxelmapContext*> m_voxel_maps;

  // the voxel lists of this context
  std::vector<CubelistContext*> m_voxel_lists;

  // the voxel maps of this context
  std::vector<CubelistContext*> m_octrees;

  // the primitive arrays of this context
  std::vector<PrimitiveArrayContext*> m_prim_arrays;

  // the dimensions of the super voxel
  Vector3ui m_dim_svoxel;

  // the minimum viewing dimension
  float m_min_view_dim;

  // the current viewing dimension
  glm::vec3 m_dim_view;

  // the maximum dimension of all voxel maps
  Vector3ui m_max_voxelmap_dim;

  // the position of the start and end voxel for the view in the voxel map
  // careful: the view will actually start at m_view_start_voxel_pos and end at (m_view_end_voxel_pos + Vector3ui(-1, -1, -1)) !
  Vector3ui m_view_start_voxel_pos, m_view_end_voxel_pos;

  // the maximum and minimum x, y and z value that shell be drawn
  Vector3ui m_max_xyz_to_draw, m_min_xyz_to_draw;

  // the interpolation color repeats after "x" voxels
  float m_interpolation_length;

  //the VBO for the grid
  GLuint m_grid_vbo;

  //the VBO for the focus point
  GLuint m_focus_point_vbo;

  //the distance between two grid lines
  float m_grid_distance;

  //the height where the grid will be drawn
  float m_grid_height;

  // the maximum grid x value
  float m_grid_max_x;

  // the maximum grid x value
  float m_grid_max_y;

  // determines what is to be drawn
  bool m_draw_edges_of_triangels;
  bool m_draw_filled_triangles;
  bool m_draw_whole_map;
  bool m_draw_grid;
  bool m_draw_collison_depth_test_always;

  // determines if the camera target should be set with content from shared memory
  bool m_getCamTargetFromShrMem;

  // the camera for this context
  gpu_voxels::visualization::Camera_gpu* m_camera;

  // color
  glm::vec4 m_background_color, m_edge_color, m_grid_color;

  //  i-th byte represents if type i should be drawn
  thrust::host_vector<uint8_t> m_draw_types;
  thrust::device_vector<uint8_t> m_d_draw_types;
  //BitVector<MAX_DRAW_TYPES> m_meanings_to_draw; // <== This should replace the draw_types
  // stores the segment position for each type
  thrust::host_vector<uint8_t> m_prefixes;
  thrust::device_vector<uint8_t> m_d_prefixes;

  // the scale factor for one unit of length and the unit type
  std::pair<float, std::string> m_scale_unit;

  // enables lighting
  bool m_lighting;
  //the intensity of the light source
  float m_light_intensity;

  int m_slice_axis;

  int m_slice_axis_position;

  uint8_t m_distance_drawmode;
};

} // end of namespace visualization
} //end of namespace gpu_voxels

#endif
