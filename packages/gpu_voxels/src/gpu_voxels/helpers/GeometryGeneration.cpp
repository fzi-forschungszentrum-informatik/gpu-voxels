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
//----------------------------------------------------------------------

#include <gpu_voxels/helpers/GeometryGeneration.h>

namespace gpu_voxels
{
namespace geometry_generation
{


void createOrientedBoxEdges(const OrientedBoxParams& params, float spacing, PointCloud &ret)
{
    std::vector<Vector3f> cloud;

    for(float x_dim = -params.dim.x; x_dim <= params.dim.x; x_dim += spacing)
    {
      cloud.push_back(Vector3f(x_dim,  params.dim.y,  params.dim.z));
      cloud.push_back(Vector3f(x_dim,  params.dim.y, -params.dim.z));
      cloud.push_back(Vector3f(x_dim, -params.dim.y,  params.dim.z));
      cloud.push_back(Vector3f(x_dim, -params.dim.y, -params.dim.z));
    }
    for(float y_dim =-params.dim.y; y_dim <= params.dim.y; y_dim += spacing)
    {
      cloud.push_back(Vector3f( params.dim.x, y_dim,  params.dim.z));
      cloud.push_back(Vector3f( params.dim.x, y_dim, -params.dim.z));
      cloud.push_back(Vector3f(-params.dim.x, y_dim,  params.dim.z));
      cloud.push_back(Vector3f(-params.dim.x, y_dim, -params.dim.z));
    }
    for(float z_dim = -params.dim.z; z_dim <= params.dim.z; z_dim += spacing)
    {
      cloud.push_back(Vector3f( params.dim.x,  params.dim.y, z_dim));
      cloud.push_back(Vector3f( params.dim.x, -params.dim.y, z_dim));
      cloud.push_back(Vector3f(-params.dim.x,  params.dim.y, z_dim));
      cloud.push_back(Vector3f(-params.dim.x, -params.dim.y, z_dim));
    }

    ret.update(cloud);

    Matrix4f transformation = Matrix4f::createFromRotationAndTranslation(Matrix3f::createFromRPY(params.rot), params.center);
    ret.transformSelf(&transformation);

    return;
}

void createOrientedBox(const OrientedBoxParams& params, float spacing, PointCloud &ret)
{
  std::vector<Vector3f> cloud;

  for(float x_dim = -params.dim.x; x_dim <= params.dim.x; x_dim += spacing)
  {
    for(float y_dim = -params.dim.y; y_dim <= params.dim.y; y_dim += spacing)
    {
      for(float z_dim = -params.dim.z; z_dim <= params.dim.z; z_dim += spacing)
      {
        Vector3f point(x_dim, y_dim, z_dim);
        cloud.push_back(point);
      }
    }
  }

  ret.update(cloud);
  Matrix4f transformation = Matrix4f::createFromRotationAndTranslation(Matrix3f::createFromRPY(params.rot), params.center);
  ret.transformSelf(&transformation);

  return;
}


std::vector<Vector3f> createBoxOfPoints(Vector3f min, Vector3f max, float delta)
{
  std::vector<Vector3f> box_cloud;
  for(float x_dim = min.x; x_dim <= max.x; x_dim += delta)
  {
    for(float y_dim = min.y; y_dim <= max.y; y_dim += delta)
    {
      for(float z_dim = min.z; z_dim <= max.z; z_dim += delta)
      {
        Vector3f point(x_dim, y_dim, z_dim);
        box_cloud.push_back(point);
      }
    }
  }
  return box_cloud;
}

std::vector<Vector3ui> createBoxOfPoints(Vector3f min, Vector3f max, float delta, float voxel_side_length)
{
  Vector3f minCroped(min.x >= 0.0f ? min.x : 0.0f, min.y >= 0.0f ? min.y : 0.0f, min.z >= 0.0f ? min.z : 0.0f );
  Vector3ui minimum(floor(minCroped.x / voxel_side_length), floor(minCroped.y / voxel_side_length), floor(minCroped.z / voxel_side_length));
  Vector3ui maximum(ceil(max.x / voxel_side_length), ceil(max.y / voxel_side_length), ceil(max.z / voxel_side_length));
  uint32_t d = round(delta / voxel_side_length);

  std::vector<Vector3ui> box_coordinates;
  for(float x_dim = minimum.x; x_dim <= maximum.x; x_dim += d)
  {
    for(float y_dim = minimum.y; y_dim <= maximum.y; y_dim += d)
    {
      for(float z_dim = minimum.z; z_dim <= maximum.z; z_dim += d)
      {
        Vector3ui point(x_dim, y_dim, z_dim);
        box_coordinates.push_back(point);
      }
    }
  }
  return box_coordinates;
}


std::vector<Vector3f> createSphereOfPoints(Vector3f center, float radius, float delta)
{
  std::vector<Vector3f> sphere_cloud;
  Vector3f bbox_min(center - Vector3f(radius));
  Vector3f bbox_max(center + Vector3f(radius));
  Vector3f point;

  for(float x_dim = bbox_min.x; x_dim <= bbox_max.x; x_dim += delta)
  {
    for(float y_dim = bbox_min.y; y_dim <= bbox_max.y; y_dim += delta)
    {
      for(float z_dim = bbox_min.z; z_dim <= bbox_max.z; z_dim += delta)
      {
        point = Vector3f(x_dim, y_dim, z_dim);

        if((center - point).length() <= radius)
        {
          sphere_cloud.push_back(point);
        }
      }
    }
  }
  return sphere_cloud;
}

std::vector<Vector3ui> createSphereOfPoints(Vector3f center, float radius, float delta, float voxel_side_length)
{
  std::vector<Vector3ui> sphere_coordinates;
  Vector3f bbox_min(center - Vector3f(radius));
  Vector3f bbox_max(center + Vector3f(radius));
  Vector3f point;

  for(float x_dim = bbox_min.x; x_dim <= bbox_max.x; x_dim += delta)
  {
    for(float y_dim = bbox_min.y; y_dim <= bbox_max.y; y_dim += delta)
    {
      for(float z_dim = bbox_min.z; z_dim <= bbox_max.z; z_dim += delta)
      {
        point = Vector3f(x_dim, y_dim, z_dim);

        if((center - point).length() <= radius)
        {
          Vector3ui coordinates = Vector3ui(round(point.x / voxel_side_length), round(point.y / voxel_side_length), round(point.z / voxel_side_length));
          sphere_coordinates.push_back(coordinates);
        }
      }
    }
  }
  return sphere_coordinates;
}

std::vector<Vector3f> createCylinderOfPoints(Vector3f center, float radius, float length_along_z, float delta)
{
  std::vector<Vector3f> cylinder_cloud;
  Vector3f bbox_min(center - Vector3f(radius, radius, length_along_z / 2.0));
  Vector3f bbox_max(center + Vector3f(radius, radius, length_along_z / 2.0));
  Vector3f point;

  for(float x_dim = bbox_min.x; x_dim <= bbox_max.x; x_dim += delta)
  {
    for(float y_dim = bbox_min.y; y_dim <= bbox_max.y; y_dim += delta)
    {
      for(float z_dim = bbox_min.z; z_dim <= bbox_max.z; z_dim += delta)
      {
        point = Vector3f(x_dim, y_dim, z_dim);

        if( sqrt ((center.x - point.x) * (center.x - point.x) +
                  (center.y - point.y) * (center.y - point.y)) <= radius )
        {
          cylinder_cloud.push_back(point);
        }
      }
    }
  }
  return cylinder_cloud;
}

std::vector<Vector3ui> createCylinderOfPoints(Vector3f center, float radius, float length_along_z, float delta, float voxel_side_length)
{
  Vector3f r(radius / voxel_side_length, radius / voxel_side_length, length_along_z / voxel_side_length / 2.0);
  Vector3f centerScaled(center.x / voxel_side_length, center.y / voxel_side_length, center.z / voxel_side_length);
  Vector3ui minimum(floor(centerScaled.x - r.x), floor(centerScaled.y - r.y), floor(centerScaled.z - r.z));
  Vector3ui maximum(ceil(centerScaled.x + r.x), ceil(centerScaled.y + r.y), ceil(centerScaled.z + r.z));
  uint32_t d = round(delta / voxel_side_length);

  Vector3ui centerCoords(round(centerScaled.x), round(centerScaled.y), round(centerScaled.z));
  std::vector<Vector3ui> cylinder_coordinates;
  Vector3ui point;
  for(float x_dim = minimum.x; x_dim <= maximum.x; x_dim += d)
  {
    for(float y_dim = minimum.y; y_dim <= maximum.y; y_dim += d)
    {
      for(float z_dim = minimum.z; z_dim <= maximum.z; z_dim += d)
      {
        point = Vector3ui(x_dim, y_dim, z_dim);

        if( sqrt ((centerCoords.x - point.x) * (centerCoords.x - point.x) +
                  (centerCoords.y - point.y) * (centerCoords.y - point.y)) <= (radius / voxel_side_length) )
        {
          cylinder_coordinates.push_back(point);
        }
      }
    }
  }
  return cylinder_coordinates;
}

void createEquidistantPointsInBox(const size_t max_nr_points,
                                  const Vector3ui max_coords,
                                  const float side_length,
                                  std::vector<Vector3f> &points)
{
  uint32_t num_points = 0;
  float x, y, z;
  for (uint32_t i = 0; i < (max_coords.x - 1) / 2; i++)
  {
    for (uint32_t j = 0; j < (max_coords.y - 1) / 2; j++)
    {
      for (uint32_t k = 0; k < (max_coords.z - 1) / 2; k++)
      {
        if (num_points >= max_nr_points)
        {
          goto OUT_OF_LOOP;
        }
        x = i * 2 * side_length + side_length / 2.0;
        y = j * 2 * side_length + side_length / 2.0;
        z = k * 2 * side_length + side_length / 2.0;
        points.push_back(Vector3f(x, y, z));
        num_points++;
      }
    }
  }
  OUT_OF_LOOP:
  return;
}


void createNonOverlapping3dCheckerboard(const size_t max_nr_points,
                                        const Vector3ui max_coords,
                                        const float side_length,
                                        std::vector<Vector3f> &black_points,
                                        std::vector<Vector3f> &white_points)
{
  uint32_t num_points = 0;
  float x, y, z;
  for (uint32_t i = 0; i < (max_coords.x - 1) / 2; i++)
  {
    for (uint32_t j = 0; j < (max_coords.y - 1) / 2; j++)
    {
      for (uint32_t k = 0; k < (max_coords.z - 1) / 2; k++)
      {
        if (num_points >= max_nr_points)
        {
          goto OUT_OF_LOOP;
        }
        x = i * 2 * side_length + side_length / 2.0;
        y = j * 2 * side_length + side_length / 2.0;
        z = k * 2 * side_length + side_length / 2.0;
        black_points.push_back(Vector3f(x, y, z));
        x = (i * 2 + 1) * side_length + side_length / 2.0;
        y = (j * 2 + 1) * side_length + side_length / 2.0;
        z = (k * 2 + 1) * side_length + side_length / 2.0;
        white_points.push_back(Vector3f(x, y, z));
        num_points++;
      }
    }
  }
  OUT_OF_LOOP:
  return;
}


} // END OF NS gpu_voxels
} // END OF NS geometry_generation
