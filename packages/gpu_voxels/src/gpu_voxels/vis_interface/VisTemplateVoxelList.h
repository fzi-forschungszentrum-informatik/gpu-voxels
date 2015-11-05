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
* \date    2015-06-10
*
*/
//----------------------------------------------------------------------

#ifndef VISTEMPLATEVOXELLIST_H
#define VISTEMPLATEVOXELLIST_H

#include <gpu_voxels/vis_interface/VisProvider.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/voxellist/TemplateVoxelList.h>

namespace gpu_voxels {

template <class Voxel, typename VoxelIDType>
class VisTemplateVoxelList : public VisProvider
{
public:
  VisTemplateVoxelList(voxellist::TemplateVoxelList<Voxel, VoxelIDType>* voxellist, std::string map_name);

  virtual ~VisTemplateVoxelList();

  virtual bool visualize(const bool force_repaint = true);

  virtual uint32_t getResolutionLevel();

protected:
  voxellist::TemplateVoxelList<Voxel, VoxelIDType>* m_voxellist;
  cudaIpcMemHandle_t* m_shm_memHandle;
  thrust::device_vector<Cube>* m_dev_buffer_1;
  thrust::device_vector<Cube>* m_dev_buffer_2;
  bool* m_shm_bufferSwapped;
  uint32_t* m_shm_num_cubes;
  bool m_internal_buffer_1;
  MapType* m_shm_voxellist_type;
};

} // namespace gpu_voxels
#endif // VISTEMPLATEVOXELLIST_H

