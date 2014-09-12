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
 * \date    2014-06-12
 *
 */
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_HELPERS_PCD_HANDLING_H_INCLUDED
#define GPU_VOXELS_HELPERS_PCD_HANDLING_H_INCLUDED

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>

/**
 * @namespace gpu_voxels::pcd_handling
 * Parser for PCD files
 */
namespace gpu_voxels {
namespace pcd_handling {

  /*!
   * \brief loadPointCloud loads a PCD file and returns the points in a vector.
   * \param path Filename
   * \param points points are written into this vector
   * \param shift_to_zero If true, the pointcloud is shifted, so its minimum coordinates lie at zero
   * \param offset_XYZ Additional transformation offset
   * \return true if succeeded, false otherwise
   */
  bool loadPointCloud(const std::string filename, std::vector<Vector3f> &points, const bool shift_to_zero=false, const Vector3f &offset_XYZ=Vector3f());

}  // end of namespace
}  // end of namespace
#endif
