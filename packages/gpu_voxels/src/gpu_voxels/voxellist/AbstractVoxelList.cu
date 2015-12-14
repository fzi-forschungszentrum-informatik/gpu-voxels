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

#include "AbstractVoxelList.h"
#include <gpu_voxels/logging/logging_voxellist.h>
#include <gpu_voxels/helpers/common_defines.h>

/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
namespace voxellist {

bool AbstractVoxelList::needsRebuild() const
{
  LOGGING_ERROR_C(VoxellistLog, AbstractVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}

bool AbstractVoxelList::rebuild()
{
  LOGGING_ERROR_C(VoxellistLog, AbstractVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}
} // end of namespace voxelmap
} // end of namespace gpu_voxels
