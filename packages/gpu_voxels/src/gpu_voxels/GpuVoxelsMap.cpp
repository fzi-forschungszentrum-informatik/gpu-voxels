
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
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

namespace gpu_voxels {

GpuVoxelsMap::GpuVoxelsMap()
{
}
GpuVoxelsMap::~GpuVoxelsMap()
{
}

bool GpuVoxelsMap::insertPointCloudFromFile(const std::string path, const bool use_model_path, const BitVoxelMeaning voxel_meaning,
                                            const bool shift_to_zero, const Vector3f &offset_XYZ, const float scaling)
{
  //load the points into the vector
  std::vector<Vector3f> points;

  if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(path, use_model_path, points, shift_to_zero, offset_XYZ, scaling))
  {
    insertPointCloud(points, voxel_meaning);
    return true;
  }
  return false;
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

MapType GpuVoxelsMap::getMapType() const
{
  return m_map_type;
}

} // end of ns

