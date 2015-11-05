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
 * \author  Florian Drews
 * \date    2013-11-07
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_POINTCLOUD_H_INCLUDED
#define GPU_VOXELS_OCTREE_POINTCLOUD_H_INCLUDED

#include <limits.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Sensor.h>
#include <gpu_voxels/octree/Voxel.h>

#include <gpu_voxels/logging/logging_octree.h>

namespace gpu_voxels {
namespace NTree {

static const DepthData INVALID_DEPTH_DATA = MAX_VALUE(DepthData);
static const DepthData MAX_DETPTH_VALUE = MAX_VALUE(DepthData);

// Transforms the point cloud output of a sensor into an sorted set of voxel without duplicates
OctreeVoxelID transformKinectPointCloud(gpu_voxels::Vector3f* point_cloud, voxel_count num_points,
                                  thrust::device_vector<Voxel>& voxel, Sensor & sensor,
                                  gpu_voxels::Vector3f voxel_dimension);

voxel_count transformKinectPointCloud_simple(gpu_voxels::Vector3f* d_point_cloud, const voxel_count num_points,
                                         thrust::device_vector<Voxel>& d_voxel, Sensor* d_sensor,
                                         const uint32_t resolution);

struct uint3comp
{
  uint32_t x, y, z;

  __host__ __device__
  friend bool operator<(uint3comp a, uint3comp b)
  {
    return ((a.x < b.x) || (a.x == b.x && a.y < b.y) || (a.x == b.x && a.y == b.y && a.z < b.z));
  }

  __host__ __device__
  friend bool operator==(uint3comp a, uint3comp b)
  {
    return (a.x == b.x && a.y == b.y && a.z == b.z);
  }
};

Vector3ui getMapDimensions(std::vector<Vector3f>& point_cloud, Vector3f& offset, float scaling = 1000.0f);

void transformPointCloud(std::vector<Vector3f>& point_cloud, std::vector<Vector3ui>& points,
                         Vector3ui& map_dimensions, float scaling = 1000.0f);

void transformDepthImage(DepthData* h_depth_image, const uint32_t width, const uint32_t height,
                         const float constant_x, const float constant_y, const float centerX,
                         const float centerY, gpu_voxels::Vector3f* d_point_cloud);
//
//void preprocessDepthImage(DepthData* d_depth_image, const uint32_t width, const uint32_t height,
//                          const DepthData noSampleValue, const DepthData shadowValue,
//                          const DepthData max_sensor_distance = MAX_RANGE);
//
//void preprocessObjectDepthImage(thrust::device_vector<DepthData>& d_depth_image, const uint32_t width,
//                            const uint32_t height, const DepthData noSampleValue, const DepthData shadowValue,
//                            const DepthData max_sensor_distance = MAX_RANGE);
//
//void preprocessFreeSpaceDepthImage(thrust::device_vector<DepthData>& d_depth_image, const uint32_t width, const uint32_t height,
//                          const DepthData noSampleValue, const DepthData shadowValue,
//                          const DepthData max_sensor_distance = MAX_RANGE);

void removeInvalidPoints(thrust::device_vector<gpu_voxels::Vector3f>& d_depth_image);

} // end of ns
} // end of ns
#endif /* POINTCLOUD_H_ */
