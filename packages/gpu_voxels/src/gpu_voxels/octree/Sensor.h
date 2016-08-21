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
 * \date    2013-11-18
 *
 */
//----------------------------------------------------------------------/*

#ifndef GPU_VOXELS_OCTREE_SENSOR_H_INCLUDED
#define GPU_VOXELS_OCTREE_SENSOR_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/SensorModel.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/Voxel.h>
#include <float.h>
#include <thrust/device_vector.h>
#include "Nodes.h"

#include <gpu_voxels/logging/logging_octree.h>

namespace gpu_voxels {
namespace NTree {

typedef uint16_t DepthData;

struct SensorDataProcessing
{
public:
  __host__ __device__
  SensorDataProcessing() :
      m_sensor_range(MAX_VALUE(DepthData)),
      m_cut_x_boarder(0),
      m_cut_y_boarder(0),
      m_remove_max_range_data(true),
      m_use_invalid_measures(false),
      m_invalid_measure(0),
      m_process_data(true),
      m_initial_probability(0),
      m_update_probability(0),
      m_voxel_side_length(0)
  {

  }

  __host__ __device__
  bool equals(const SensorDataProcessing& b) const
  {
    return this->m_sensor_range == b.m_sensor_range
        && this->m_cut_x_boarder == b.m_cut_x_boarder
        && this->m_cut_y_boarder == b.m_cut_y_boarder
        && this->m_remove_max_range_data == b.m_remove_max_range_data
        && this->m_use_invalid_measures == b.m_use_invalid_measures
        && this->m_invalid_measure == b.m_invalid_measure
        && this->m_process_data == b.m_process_data
        && this->m_initial_probability == b.m_initial_probability
        && this->m_update_probability == b.m_update_probability
        && this->m_voxel_side_length == b.m_voxel_side_length;
  }

  DepthData m_sensor_range;
  int m_cut_x_boarder;
  int m_cut_y_boarder;
  bool m_remove_max_range_data;
  bool m_use_invalid_measures;
  DepthData m_invalid_measure;
  bool m_process_data;
  Probability m_initial_probability;
  Probability m_update_probability;
  int m_voxel_side_length; // voxel side length in mm
};

//struct ProcessedSensorData
//{
//public:
//  __host__ __device__
//  ProcessedSensorData()
//  {
//
//  }
//
//  thrust::device_vector<Voxel> free_space_voxel;
//  thrust::device_vector<Voxel> object_voxel;
//};

struct Sensor
{
public:

  __host__ __device__
  Sensor()
  {
  }

  __host__ __device__
  Sensor(gpu_voxels::Matrix4f _pose, uint32_t _data_width, uint32_t _data_height) :
      pose(_pose), data_width(_data_width), data_height(_data_height)
  {
  }

  __host__ __device__
  Sensor(const Sensor& other) :
      pose(other.pose), data_width(other.data_width), data_height(
          other.data_height)
  {
  }

  /**
   * Takes coordinates in meter and returns the transformed coordinates also in meter
   */
  __host__ __device__
  __forceinline__
  gpu_voxels::Vector3f sensorCoordinatesToWorldCoordinates(const gpu_voxels::Vector3f& point)
  {
    return pose * point;
  }

  /*
   * Software License Agreement (BSD License)
   *
   * Point Cloud Library (PCL) - www.pointclouds.org
   * Copyright (c) 2010-2011, Willow Garage, Inc.
   * Copyright (c) 2012-, Open Perception, Inc.
   *
   * All rights reserved.
   *
   * Redistribution and use in source and binary forms, with or without
   * modification, are permitted provided that the following conditions
   * are met:
   *
   * * Redistributions of source code must retain the above copyright
   * notice, this list of conditions and the following disclaimer.
   * * Redistributions in binary form must reproduce the above
   * copyright notice, this list of conditions and the following
   * disclaimer in the documentation and/or other materials provided
   * with the distribution.
   * * Neither the name of the copyright holder(s) nor the names of its
   * contributors may be used to endorse or promote products derived
   * from this software without specific prior written permission.
   *
   * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   * POSSIBILITY OF SUCH DAMAGE.
   *
   */
  /**
   * Returned coordinates are in meter.
   */
  __host__ __device__
  __forceinline__
  gpu_voxels::Vector3f sensorMeasureToSensorCoordinates(DepthData measure, const int x, const int y,
                                                 const DepthData invalid_measure)
  {
    // Intrinsic parameter of the Microsoft Kinect
    const float constant_x = 0.001753, constant_y = 0.001753, centerX = 319.500000, centerY = 239.500000;

    gpu_voxels::Vector3f my_point;
    if (measure == invalid_measure)
      my_point = gpu_voxels::Vector3f(NAN, NAN, NAN);
    else
    {
      my_point.z = measure * 0.001f;
      my_point.x = (float(x) - centerX) * my_point.z * constant_x;
      my_point.y = (float(y) - centerY) * my_point.z * constant_y;
    }
    return my_point;
  }

  __host__
  void processSensorData(const DepthData* h_sensor_data,
                         thrust::device_vector<Voxel> *&d_free_space_voxel,
                         thrust::device_vector<Voxel> *&d_object_voxel);

  __host__
  void processSensorData(const Vector3f *h_points,
                                 thrust::device_vector<Voxel> *&d_free_space_voxel,
                                 thrust::device_vector<Voxel> *&d_object_voxel);

private:
  __host__
  void _processDepthImage(const DepthData* h_sensor_data,
                                 thrust::device_vector<Vector3f>& d_free_space_points,
                                 thrust::device_vector<Vector3f>& d_object_points);
  __host__
  void _processSensorData(thrust::device_vector<Vector3f>& d_free_space_points,
                                 thrust::device_vector<Vector3f>& d_object_points,
                                 thrust::device_vector<Voxel>& d_free_space_voxel,
                                 thrust::device_vector<Voxel>& d_object_voxel);

public:
  //gpu_voxels::Vector3f position;
  //gpu_voxels::Matrix4f orientation;

  gpu_voxels::Matrix4f pose;

  uint32_t data_width;
  uint32_t data_height;
  SensorDataProcessing object_data;
  SensorDataProcessing free_space_data;

  SensorModel sensorModel;
};

} // end of ns
} // end of ns

#endif /* SENSOR_H_ */
