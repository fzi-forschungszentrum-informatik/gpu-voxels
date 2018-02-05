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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-06-10
*
*/
//----------------------------------------------------------------------

#include <gpu_voxels/vis_interface/VisTemplateVoxelList.hpp>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {
  template class VisTemplateVoxelList<BitVectorVoxel, uint32_t>;
  template class VisTemplateVoxelList<CountingVoxel, uint32_t>;
} // namespace gpu_voxels
