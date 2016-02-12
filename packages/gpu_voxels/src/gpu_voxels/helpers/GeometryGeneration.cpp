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
 * \author  Andreas Hermann
 * \date    2016-01-09
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/helpers/GeometryGeneration.h>

namespace gpu_voxels
{
namespace geometry_generation
{


void createOrientedBoxEdges(const OrientedBoxParams& params, float spacing, gpu_voxels::MetaPointCloud &ret)
{
    std::vector<gpu_voxels::Vector3f> cloud;

    for(float x_dim = -params.dim.x; x_dim <= params.dim.x; x_dim += spacing)
    {
      cloud.push_back(Vector3f(x_dim,  params.dim.y,  params.dim.z));
      cloud.push_back(Vector3f(x_dim,  params.dim.y, -params.dim.z));
      cloud.push_back(Vector3f(x_dim, -params.dim.y,  params.dim.z));
      cloud.push_back(Vector3f(x_dim, -params.dim.y, -params.dim.z));
    }
    for(float y_dim =-params.dim.y; y_dim <= params.dim.y; y_dim += spacing)
    {
      cloud.push_back(Vector3f( params.dim.x, y_dim,  params.dim.z));
      cloud.push_back(Vector3f( params.dim.x, y_dim, -params.dim.z));
      cloud.push_back(Vector3f(-params.dim.x, y_dim,  params.dim.z));
      cloud.push_back(Vector3f(-params.dim.x, y_dim, -params.dim.z));
    }
    for(float z_dim = -params.dim.z; z_dim <= params.dim.z; z_dim += spacing)
    {
      cloud.push_back(Vector3f( params.dim.x,  params.dim.y, z_dim));
      cloud.push_back(Vector3f( params.dim.x, -params.dim.y, z_dim));
      cloud.push_back(Vector3f(-params.dim.x,  params.dim.y, z_dim));
      cloud.push_back(Vector3f(-params.dim.x, -params.dim.y, z_dim));
    }

    ret.updatePointCloud(0, cloud, true);
    gpu_voxels::Matrix4f transformation;
    transformation = gpu_voxels::rotateYPR(params.rot);
    gpu_voxels::Vec3ToMat4(params.center, transformation);

    ret.transformSelf(&transformation);

    return;
}

void createOrientedBox(const OrientedBoxParams& params, float spacing, gpu_voxels::MetaPointCloud &ret)
{
  std::vector<gpu_voxels::Vector3f> cloud;

  for(float x_dim = -params.dim.x; x_dim <= params.dim.x; x_dim += spacing)
  {
    for(float y_dim =-params.dim.y; y_dim <= params.dim.y; y_dim += spacing)
    {
      for(float z_dim = -params.dim.z; z_dim <= -params.dim.z; z_dim += spacing)
      {
        Vector3f point(x_dim, y_dim, z_dim);
        cloud.push_back(point);
      }
    }
  }

  ret.updatePointCloud(0, cloud, true);
  gpu_voxels::Matrix4f transformation;
  transformation = gpu_voxels::rotateYPR(params.rot);
  gpu_voxels::Vec3ToMat4(params.center, transformation);

  ret.transformSelf(&transformation);

  return;
}


} // END OF NS gpu_voxels
} // END OF NS geometry_generation
