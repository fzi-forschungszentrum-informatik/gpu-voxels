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
//----------------------------------------------------------------------/*

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/CudaMath.h>

namespace gpu_voxels
{
namespace geometry_generation
{

/*!
 * \brief createOrientedBoxEdges creates points on the edges of a box
 * \param params specify the box with center, half the side length and orientation
 * \param spacing defines sampling distance for points along the edges
 * \param ret MetaPointCloud whose first cloud is written into
 */
void createOrientedBoxEdges(const OrientedBoxParams& params, float spacing, gpu_voxels::MetaPointCloud &ret);

/*!
 * \brief createOrientedBox creates a dense solid cloud of a box
 * \param params specify the box with center, half the side length and orientation
 * \param spacing defines sampling distance for points in the volume
 * \param ret MetaPointCloud whose first cloud is written into
 */
void createOrientedBox(const OrientedBoxParams& params, float spacing, gpu_voxels::MetaPointCloud &ret);


// TODO: Move helper functions from Boost-Tests to here.

} // END OF NS gpu_voxels
} // END OF NS geometry_generation
