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
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VISVOXELLIST_H
#define GPU_VOXELS_VISVOXELLIST_H

#include <gpu_voxels/vis_interface/VisProvider.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/voxellist/AbstractVoxelList.h>

namespace gpu_voxels {

class VisVoxelList : public VisProvider
{
public:
  VisVoxelList(voxellist::AbstractVoxelList* voxellist, std::string map_name);

  virtual ~VisVoxelList();

  virtual bool visualize(const bool force_repaint = true);

  virtual uint32_t getResolutionLevel();

protected:
  voxellist::AbstractVoxelList* m_voxellist;
  cudaIpcMemHandle_t* m_shm_memHandle;
  uint32_t* m_shm_list_size;
  float* m_shm_VoxelSize;
  MapType* m_shm_voxellist_type;
  bool* m_shm_voxellist_changed;
};

}

#endif // GPU_VOXELS_VISVOXELLIST_H
