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
 * \date    2014-05-16
 *
 */
//----------------------------------------------------------------------

#ifndef OCTOMAPPROVIDER_H_
#define OCTOMAPPROVIDER_H_


#include <octomap/octomap.h>
#include <gpu_voxels/octree/test/Provider.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>

namespace gpu_voxels {
namespace NTree {
namespace Provider {

class OctomapProvider: public Provider
{
public:

  OctomapProvider();

  virtual ~OctomapProvider();

  virtual void visualize();

  virtual void init(Provider_Parameter& parameter);

  virtual void newSensorData(const DepthData* h_depth_data, const uint32_t width, const uint32_t height);


  virtual void newSensorData(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points, const uint32_t width,
                               const uint32_t height);

  virtual void collide();

  virtual bool waitForNewData(volatile bool* stop);

protected:
  virtual octomap::Pointcloud* toOctoPointCloud(gpu_voxels::Vector3f* h_point_cloud, const uint32_t num_points);


  octomap::OcTree* m_octree;


};

}
}
}

#endif /* OCTOMAPPROVIDER_H_ */
