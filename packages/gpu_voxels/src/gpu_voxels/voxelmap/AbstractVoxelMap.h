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
 * \date    2014-07-10
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_ABSTRACT_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_ABSTRACT_VOXELMAP_H_INCLUDED

#include <gpu_voxels/GpuVoxelsMap.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/logging/logging_voxelmap.h>

#include <vector>
#include <string>

/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
namespace voxelmap {

class AbstractVoxelMap : public GpuVoxelsMap
{
public:

  //! get pointer to data array on device
  virtual void* getVoidDeviceDataPtr() = 0;

  //! get the side length of the voxels.
  virtual float getVoxelSideLength() const = 0;

  virtual void insertPointCloud(const std::vector<Vector3f> &points, const BitVoxelMeaning voxel_meaning) = 0;

  virtual void insertPointCloud(const PointCloud &pointcloud, const BitVoxelMeaning voxel_meaning) = 0;

  virtual void insertPointCloud(const Vector3f* points_d, uint32_t size, const BitVoxelMeaning voxel_meaning) = 0;

  //! get the number of bytes that is required for the voxelmap
  virtual size_t getMemoryUsage() const = 0;

  virtual MapType getTemplateType() const = 0;



  // ------ BEGIN Global API functions ------
  virtual bool needsRebuild() const;

  virtual bool rebuild();
  // ------ END Global API functions ------

};

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
