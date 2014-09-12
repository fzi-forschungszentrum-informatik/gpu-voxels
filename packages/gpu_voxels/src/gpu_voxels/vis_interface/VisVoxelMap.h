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
#ifndef GPU_VOXELS_VISVOXELMAPPROB_H_INCLUDED
#define GPU_VOXELS_VISVOXELMAPPROB_H_INCLUDED

#include <gpu_voxels/vis_interface/VisProvider.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>
#include <gpu_voxels/voxelmap/AbstractVoxelMap.h>

#include <cuda_runtime.h>

namespace gpu_voxels {

class VisVoxelMap: public VisProvider
{
public:

  VisVoxelMap(voxelmap::AbstractVoxelMap* voxelmap, std::string map_name);

  virtual ~VisVoxelMap();

  virtual bool visualize(const bool force_repaint = true);

  virtual uint32_t getResolutionLevel();

protected:
  voxelmap::AbstractVoxelMap* m_voxelmap;
  cudaIpcMemHandle_t* m_shm_memHandle;
  Vector3ui* m_shm_mapDim;
  float* m_shm_VoxelSize;
  MapType* m_shm_voxelmap_type;
  bool* m_shm_voxelmap_changed;
};

}

#endif
