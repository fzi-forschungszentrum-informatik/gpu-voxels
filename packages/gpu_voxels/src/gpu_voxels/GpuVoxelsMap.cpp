
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
 * \date    2014-06-08
 *
 */
//----------------------------------------------------------------------

#include "GpuVoxelsMap.h"

namespace gpu_voxels {

GpuVoxelsMap::GpuVoxelsMap()
{
}
GpuVoxelsMap::~GpuVoxelsMap()
{
}

void GpuVoxelsMap::translateContent(int32_t x, int32_t y, int32_t z)
{
}

size_t GpuVoxelsMap::collideWithRelativeTransform(const GpuVoxelsMapSharedPtr other, float coll_threshold, int32_t x, int32_t y, int32_t z)
{
  return 0;
}

bool GpuVoxelsMap::insertPCD(const std::string path, VoxelType voxel_type, const bool shift_to_zero, const Vector3f &offset_XYZ)
{
  std::vector<Vector3f> points;
  //load the points into the vector
  if (!pcd_handling::loadPointCloud(path, points, shift_to_zero, offset_XYZ))
  {
    return false;
  }
  insertGlobalData(points, voxel_type);
  return true;
}

void GpuVoxelsMap::generateVisualizerData()
{
}

bool GpuVoxelsMap::rebuildIfNeeded()
{
  if(needsRebuild())
  {
    rebuild();
    return true;
  }
  else
    return false;
}

MapType GpuVoxelsMap::getMapType()
{
  return m_map_type;
}

} // end of ns

