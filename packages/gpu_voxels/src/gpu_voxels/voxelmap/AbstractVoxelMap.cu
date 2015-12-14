// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-07-12
 *
 */
//----------------------------------------------------------------------
#include "AbstractVoxelMap.h"

/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
namespace voxelmap {

bool AbstractVoxelMap::needsRebuild() const
{
  LOGGING_ERROR_C(VoxelmapLog, AbstractVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}

bool AbstractVoxelMap::rebuild()
{
  LOGGING_ERROR_C(VoxelmapLog, AbstractVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}
} // end of namespace voxelmap
} // end of namespace gpu_voxels
