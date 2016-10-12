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
 * \date    2016-11-02
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_HEIGHTMAPLOADER_H_INCLUDED
#define GPU_VOXELS_HELPERS_HEIGHTMAPLOADER_H_INCLUDED

#include <string>
#include "gpu_voxels/helpers/cuda_datatypes.h"
#include "gpu_voxels/helpers/PointCloud.h"

namespace gpu_voxels {
namespace file_handling {


void HeightMapLoader(std::string bottom_map, std::string ceiling_map, bool use_model_path,
                                 size_t bottom_start_height, size_t ceiling_end_height,
                                 float meter_per_pixel, float meter_per_greyshade,
                                 gpu_voxels::Vector3f metric_offset, PointCloud &cloud);

}  // end of namespace
}  // end of namespace

#endif // HeightMapLoader_H
