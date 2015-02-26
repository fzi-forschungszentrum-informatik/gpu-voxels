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
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_HELPERS_VIS_PROVIDER_H_INCLUDED
#define GPU_VOXELS_HELPERS_VIS_PROVIDER_H_INCLUDED
#include <gpu_voxels/helpers/CompileIssues.h>

#include <gpu_voxels/vis_interface/VisualizerInterface.h>

#include <cstdio>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/permissions.hpp>

#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>


namespace gpu_voxels {

class VisProvider;
typedef boost::shared_ptr<VisProvider> VisProviderSharedPtr;

/**
 * @brief VisProvider Superclass to handle the visualization of different map types
 * with gpu_visualization through shared memory
 */
class VisProvider
{
public:

  VisProvider(std::string segment_name, std::string map_name);

  virtual ~VisProvider();

  virtual bool visualize(const bool force_repaint = true) = 0;

  virtual uint32_t getResolutionLevel() = 0;

  virtual void setDrawTypes(DrawTypes toggle_draw_types);

protected:

  void openOrCreateSegment();

  boost::interprocess::managed_shared_memory m_segment;
  boost::interprocess::managed_shared_memory m_visualizer_segment;
  std::string m_segment_name;
  std::string m_map_name;
  char* m_shm_mapName;
  DrawTypes* m_shm_draw_types;
};

}

#endif
