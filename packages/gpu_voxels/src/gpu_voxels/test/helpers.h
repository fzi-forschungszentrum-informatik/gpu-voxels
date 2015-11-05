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
 * \date    2015-10-16
 *
 */
//----------------------------------------------------------------------/*

#ifndef GPU_VOXELS_TEST_HELPERS_H_INCLUDED
#define GPU_VOXELS_TEST_HELPERS_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>

namespace gpu_voxels {


std::vector<Vector3f> createBoxOfPoints(Vector3f min, Vector3f max, float delta);

/*!
 * \brief createEquidistantPointsInBox creates a pointcloud that will cover
 * \code max_nr_points voxels in a box with \code max_coords and a given discretization
 * of \code side_length
 * \param max_nr_points
 * \param max_coords
 * \param side_length
 * \param points
 */
void createEquidistantPointsInBox(const size_t max_nr_points,
                                  const Vector3ui max_coords,
                                  const float side_length,
                                  std::vector<Vector3f> &points);

/*!
 * \brief createNonOverlapping3dCheckerboard creates two pointclouds that will not overlap
 * when discretized with a map of sidelength \code side_length
 * \param max_nr_points
 * \param max_coords
 * \param side_length
 * \param black_points
 * \param white_points
 */
void createNonOverlapping3dCheckerboard(const size_t max_nr_points,
                                        const Vector3ui max_coords,
                                        const float side_length,
                                        std::vector<Vector3f> &black_points,
                                        std::vector<Vector3f> &white_points);

} // end of ns
#endif
