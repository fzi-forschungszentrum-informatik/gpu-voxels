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
#include <gpu_voxels/helpers/PointCloud.h>

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
void createOrientedBoxEdges(const OrientedBoxParams& params, float spacing, gpu_voxels::PointCloud &ret);

/*!
 * \brief createOrientedBox creates a dense solid cloud of a box
 * \param params specify the box with center, half the side length and orientation
 * \param spacing defines sampling distance for points in the volume
 * \param ret MetaPointCloud whose first cloud is written into
 */
void createOrientedBox(const OrientedBoxParams& params, float spacing, gpu_voxels::PointCloud &ret);

/*!
 * \brief createBoxOfPoints creates points within given bounds with a given spacing
 * \param min Lower corner of the box (lower bound) and also the coords of the first point
 * \param max Inclusive upper corner of the box (points <= upper bound)
 * \param delta Equidistant spacing (increment) between the points
 * \return Generated pointcloud
 */
std::vector<Vector3f> createBoxOfPoints(Vector3f min, Vector3f max, float delta);

std::vector<Vector3ui> createBoxOfPoints(Vector3f min, Vector3f max, float delta, float voxel_side_length);

/*!
 * \brief createSphereOfPoints creates points within a sphere, with a given spacing
 * \param center Center of the sphere
 * \param radius Radius of the sphere (points <= radius)
 * \param delta Equidistant spacing (increment) between the points
 * \return Generated pointcloud
 */
std::vector<Vector3f> createSphereOfPoints(Vector3f center, float radius, float delta);

std::vector<Vector3ui> createSphereOfPoints(Vector3f center, float radius, float delta, float voxel_side_length);

/*!
 * \brief createCylinderOfPoints creates points within a cylinder, with a given spacing
 * \param center Center of the cylinder (center of gravity)
 * \param radius Radius of the circly lying in the xy-plane
 * \param length_along_z The height of the cylinder along the z-axis
 * \param delta Equidistant spacing (increment) between the points
 * \return Generated pointcloud
 */
std::vector<Vector3f> createCylinderOfPoints(Vector3f center, float radius, float length_along_z, float delta);

std::vector<Vector3ui> createCylinderOfPoints(Vector3f center, float radius, float length_along_z, float delta, float voxel_side_length);

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

} // END OF NS gpu_voxels
} // END OF NS geometry_generation
