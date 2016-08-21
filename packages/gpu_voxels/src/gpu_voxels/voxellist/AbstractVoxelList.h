// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
*
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXELLIST_ABSTRACTVOXELLIST_H
#define GPU_VOXELS_VOXELLIST_ABSTRACTVOXELLIST_H

#include <gpu_voxels/GpuVoxelsMap.h>

namespace gpu_voxels {
namespace voxellist {


class AbstractVoxelList : public GpuVoxelsMap
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

  virtual MapType getTemplateType() = 0;

  // ------ BEGIN Global API functions ------
  virtual bool needsRebuild() const;

  virtual bool rebuild();
  // ------ END Global API functions ------
};

} // end of namespace voxellist
} // end of namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_ABSTRACTVOXELLIST_H
