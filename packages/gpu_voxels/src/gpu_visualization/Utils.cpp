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
 * \author  Matthias Wagner
 * \date    2014-02-10
 *
 */
//----------------------------------------------------------------------
#include "Utils.h"

namespace gpu_voxels {
namespace visualization {

void ExitOnGLError(const std::string error_message)
{
  const GLenum ErrorValue = glGetError();

  if (ErrorValue != GL_NO_ERROR)
  {
    LOGGING_ERROR(Visualization, error_message << ": " << (char*)gluErrorString(ErrorValue) << endl);
    exit(EXIT_FAILURE);
  }
}

std::string typeToString(VoxelType type)
{
  std::string name = "";
  switch (type)
  {
    case eVT_UNDEFINED:
      name = "UNDEFINED";
      break;
    case eVT_OCCUPIED:
      name = "OCCUPIED";
      break;
    case eVT_FREE:
      name = "FREE";
      break;
    case eVT_COLLISION:
      name = "COLLISION";
      break;
    case eVT_SWEPT_VOLUME_END:
      name = "SWEPT_VOLUME_END";
      break;
    case eVT_SWEPT_VOLUME_START:
      name = "SWEPT_VOLUME_START";
      break;
    case eVT_UNKNOWN:
      name = "UNKNOWN";
      break;
    default:
      name = "SWEPT_VOLUME";
      break;
  }
  std::ostringstream convert;
  convert << (uint32_t) type;
  name.append("(" + convert.str() + ")");
  return name;
}
std::string typeToString(MapType type)
{
  std::string name = "";
  switch (type)
  {
    case MT_BITVECTOR_VOXELLIST:
      name = "VOXELLIST";
      break;
    case MT_BITVECTOR_OCTREE:
      name = "OCTREE";
      break;
    case MT_BITVECTOR_MORTON_VOXELLIST:
      name = "OCTREE_VOXELLIST";
      break;
    case MT_PROBAB_VOXELMAP:
      name = "PROBAB_VOXELMAP";
      break;
    case MT_PROBAB_VOXELLIST:
      name = "PROBAB_VOXELLIST";
      break;
    case MT_PROBAB_MORTON_VOXELLIST:
      name = "PROBAB_OCTREE_VOXELLIST";
      break;
    case MT_BITVECTOR_VOXELMAP:
      name = "BIT_VOXELMAP";
      break;
    default:
      name = "";
      break;
  }
  return name;
}

} // end of namespace visualization
} // end of namespace gpu_voxels
