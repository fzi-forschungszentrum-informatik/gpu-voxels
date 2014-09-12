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
 * \date    2014-04-05
 *
 */
//----------------------------------------------------------------------

#ifndef VOXELMAPPROVIDER_H_
#define VOXELMAPPROVIDER_H_

#include "Provider.h"
#include <gpu_voxels/voxelmap/AbstractVoxelMap.h>

namespace gpu_voxels {
namespace NTree {
namespace Provider {

class VoxelMapProvider: public Provider
{
public:

  VoxelMapProvider();

  virtual ~VoxelMapProvider();

  virtual void visualize();

  virtual void init(Provider_Parameter& parameter);

  virtual void newSensorData(const DepthData* h_depth_data, const uint32_t width, const uint32_t height);

  virtual void newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points, const uint32_t width,
                             const uint32_t height);

  virtual void collide();

  virtual bool waitForNewData(volatile bool* stop);

  virtual gpu_voxels::voxelmap::AbstractVoxelMap* getVoxelMap();

protected:
  void collide_wo_locking();

  gpu_voxels::voxelmap::AbstractVoxelMap* m_voxelMap;
  cudaIpcMemHandle_t* m_shm_memHandle;
  Vector3ui* m_shm_mapDim;
  float* m_shm_VoxelSize;
};

}
}
}

#endif /* VOXELMAPPROVIDER_H_ */
