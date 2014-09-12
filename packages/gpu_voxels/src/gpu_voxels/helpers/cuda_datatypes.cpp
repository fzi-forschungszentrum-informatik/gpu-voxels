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
 * \author  Sebastian Klemm
 * \date    2012-06-22
 *
 */
//----------------------------------------------------------------------

#include "cuda_datatypes.h"

namespace gpu_voxels {

void centerPointCloud(std::vector<Vector3f> &points)
{
  Vector3f min_xyz = points[0];
  Vector3f max_xyz = points[0];
  Vector3f center_offset_xyz;

  for (size_t i = 1; i < points.size(); i++)
  {
    min_xyz.x = std::min(min_xyz.x, points[i].x);
    min_xyz.y = std::min(min_xyz.y, points[i].y);
    min_xyz.z = std::min(min_xyz.z, points[i].z);

    max_xyz.x = std::max(max_xyz.x, points[i].x);
    max_xyz.y = std::max(max_xyz.y, points[i].y);
    max_xyz.z = std::max(max_xyz.z, points[i].z);
  }

  center_offset_xyz.x = (min_xyz.x + max_xyz.x) / 2.0;
  center_offset_xyz.y = (min_xyz.y + max_xyz.y) / 2.0;
  center_offset_xyz.z = (min_xyz.z + max_xyz.z) / 2.0;

  for (size_t i = 0; i < points.size(); i++)
  {
    points[i].x -= center_offset_xyz.x;
    points[i].y -= center_offset_xyz.y;
    points[i].z -= center_offset_xyz.z;
  }
}

void shiftPointCloudToZero(std::vector<Vector3f> &points)
{
  Vector3f min_xyz = points[0];

  for (size_t i = 1; i < points.size(); i++)
  {
    min_xyz.x = std::min(min_xyz.x, points[i].x);
    min_xyz.y = std::min(min_xyz.y, points[i].y);
    min_xyz.z = std::min(min_xyz.z, points[i].z);
  }

  for (size_t i = 0; i < points.size(); i++)
  {
    points[i].x -= min_xyz.x;
    points[i].y -= min_xyz.y;
    points[i].z -= min_xyz.z;
  }
}

} // end of ns

