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
 * \date    2014-03-24
 *
 */
//----------------------------------------------------------------------

#include "Helper.h"

#include <gpu_voxels/helpers/cuda_handling.h>

#include <algorithm>
#include <iostream>

using namespace std;

namespace gpu_voxels {
namespace NTree {
namespace Test {

bool testAndInitDevice()
{

// Find/set the device.
// The test requires an architecture SM35 or greater (CDP capable).

  int device_count = 0, device = -1;
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  for (int i = 0; i < device_count; ++i)
  {
    cudaDeviceProp properties;
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&properties, i));
    if (properties.major > 2 || (properties.major == 2 && properties.minor >= 0))
    {
      device = i;
      //warp_size = properties.warpSize;
      std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
      break;
    }
  }
  if (device == -1)
  {
    std::cerr << "No device with SM 3.5 or higher found, which is required for CUDA Dynamic Parallelism.\n"
        << std::endl;
    return false;
  }
  cudaSetDevice(device);
  HANDLE_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
//HANDLE_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  return true;
}

thrust::host_vector<gpu_voxels::Vector3ui> linearPoints(voxel_count num_points, OctreeVoxelID maxValue)
{
  uint32_t max_coordinate = (uint32_t) ceil(pow(maxValue, 1.0 / 3));
  thrust::host_vector<gpu_voxels::Vector3ui> points(num_points);
  for (voxel_count i = 0; i < num_points; ++i)
  {
    points[i].x = (uint32_t) (i % max_coordinate);
    points[i].y = (uint32_t) ((i / max_coordinate) % max_coordinate);
    points[i].z = (uint32_t) ((i / (max_coordinate * max_coordinate)) % max_coordinate);
    //printf("%u %u %u\n", points[i].x, points[i].y, points[i].z);
  }
  return points;
}

void getRandomPlan(vector<Vector3f>& robot, vector<Vector3f>& random_plan, const int path_points,
                   Vector3f map_size, const float step_size)
{
  printf("Generating random plan\n");

  vector<Vector3f> norm_robot = robot;
  Vector3f robot_min(INT_MAX);
  Vector3f robot_max(INT_MIN);
  for (size_t i = 0; i < robot.size(); ++i)
  {
    Vector3f point = robot[i];
    robot_min.x = min(robot_min.x, point.x);
    robot_min.y = min(robot_min.y, point.y);
    robot_min.z = min(robot_min.z, point.z);
    robot_max.x = max(robot_max.x, point.x);
    robot_max.y = max(robot_max.y, point.y);
    robot_max.z = max(robot_max.z, point.z);
  }

  printf("min %f %f %f\n", robot_min.x, robot_min.y, robot_min.z);
  printf("max %f %f %f\n", robot_max.x, robot_max.y, robot_max.z);

  for (size_t i = 0; i < robot.size(); ++i)
  {
    norm_robot[i] = norm_robot[i] - robot_min;
  }

//  // generate more points by shifting the robot in all directions
//  int shifts_per_dim = 3;
//  float res = 0.04;
//  std::vector<Vector3f> tmp_rob = norm_robot;
//  int pos = 0;
//  norm_robot.resize(shifts_per_dim * shifts_per_dim * shifts_per_dim * norm_robot.size());
//  for (int x = -shifts_per_dim / 2; x <= shifts_per_dim / 2; ++x)
//  {
//    for (int y = -shifts_per_dim / 2; y <= shifts_per_dim / 2; ++y)
//    {
//      for (int z = -shifts_per_dim / 2; z <= shifts_per_dim / 2; ++z)
//      {
//        for (size_t i = 0; i < tmp_rob.size(); ++i)
//        {
//          norm_robot[pos++] = tmp_rob[i] + Vector3f(x, y, z) * res;
//        }
//      }
//    }
//  }
//  map = norm_robot;

//random_plan.resize((runs + 1) * robot.size());
  Vector3f rob_size = robot_max - robot_min;
  float robot_side_length = max(max(rob_size.x, rob_size.y), rob_size.z);
  Vector3f min_rand = Vector3f(robot_side_length);
  Vector3f max_rand = map_size - Vector3f(robot_side_length);

  Vector3f old_point;
  old_point.x = min_rand.x + (drand48() * (max_rand.x - min_rand.x));
  old_point.y = min_rand.x + (drand48() * (max_rand.y - min_rand.y));
  old_point.z = min_rand.z + (drand48() * (max_rand.z - min_rand.z));
  for (int p = 0; p < path_points; ++p)
  {
    Vector3f new_point;
    new_point.x = min_rand.x + (drand48() * (max_rand.x - min_rand.x));
    new_point.y = min_rand.x + (drand48() * (max_rand.y - min_rand.y));
    new_point.z = min_rand.z + (drand48() * (max_rand.z - min_rand.z));

    Vector3f diff = new_point - old_point;
    int steps = (int) ceil(max(max(abs(diff.x), abs(diff.y)), abs(diff.z)) / step_size);
    printf("num_steps %i\n", steps);

    int from = random_plan.size();
    int pos = from;
    random_plan.resize(random_plan.size() + steps * norm_robot.size());
    for (int s = 0; s < steps; ++s)
    {
      Vector3f offset = (diff / float(steps)) * float(s);
      for (size_t i = 0; i < norm_robot.size(); ++i)
      {
        random_plan[pos++] = norm_robot[i] + old_point + offset;
      }
    }
    old_point = new_point;

    // remove duplicates
    sort(random_plan.begin() + from, random_plan.end(), Vector3f::compVec);
    random_plan.erase(unique(random_plan.begin() + from, random_plan.end(), Vector3f::eqlVec), random_plan.end());
  }

  printf("New random_plan with %lu points\n", random_plan.size());

//// save as pcd
//  pcl::PointCloud < pcl::PointXYZ > cloud;
//  cloud.resize(random_plan.size());
//  for (int i = 0; i < random_plan.size(); ++i)
//  {
//    cloud[i].x = random_plan[i].x;
//    cloud[i].y = random_plan[i].y;
//    cloud[i].z = random_plan[i].z;
//  }
//  std::string t = getTime_str();
//  const std::string filename = "./RobotPlan_" + t + ".pcd";
//  pcl::io::savePCDFileASCII < pcl::PointXYZ > (filename, cloud);
//
//  printf("New random_plan '%s' with %lu points\n", filename.c_str(), cloud.size());
}

void getRandomPlans(vector<Provider::Provider_Parameter>& parameter)
{
  for (uint32_t i = 0; i < parameter.size(); ++i)
  {
    if (parameter[i].mode == Provider::Provider_Parameter::MODE_RANDOM_PLAN)
    {
      if (parameter[i].points.size() < 1)
        printf("Robot model missing!\n");
      else
      {
        vector<Vector3f> random_plan;
        Vector3f map_size;
        getRandomPlan(parameter[i].points, random_plan, 5, parameter[i].plan_size);
        parameter[i].points = random_plan;
      }
    }
  }
}

}
}
}

