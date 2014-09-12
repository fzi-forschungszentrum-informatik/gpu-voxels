// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner <mwagner@fzi.de>
 * \date    2014-06-13
 *
 */
//----------------------------------------------------------------------
#include "VoxelMap.h"
#include "kernels/VoxelMapOperations.h"

namespace gpu_voxels {
namespace voxelmap {
//
//bool VoxelMap::triggerVoxelMapAddresSchemeTest(uint32_t nr_of_tests)
//{
//  Vector3f testpoint;
//  Vector3f * dev_testpoint;
//  srand(time(0));
//
//  bool success;
//  bool* dev_success;
//  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_success, sizeof(bool)));
//  HANDLE_CUDA_ERROR(cudaMalloc((void** )&dev_testpoint, sizeof(Vector3f)));
//
//  for (uint32_t i = 0; i < nr_of_tests; i++)
//  {
//    testpoint.x = (rand() / (double) RAND_MAX) * m_limits.x;
//    testpoint.y = (rand() / (double) RAND_MAX) * m_limits.y;
//    testpoint.z = (rand() / (double) RAND_MAX) * m_limits.z;
//
//    HANDLE_CUDA_ERROR(cudaMemcpy(dev_testpoint, &testpoint, sizeof(Vector3f), cudaMemcpyHostToDevice));
//
//    voxelmap::kernelAddressingTest<<< 1, 1 >>>
//    (m_dev_data, m_dev_dim, m_dev_voxel_side_length, dev_testpoint, dev_success);
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//
//    HANDLE_CUDA_ERROR(cudaMemcpy(&success, dev_success, sizeof(bool), cudaMemcpyDeviceToHost));
//    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//    if (!success)
//    {
//      return false;
//    }
//  }
//  return true;
//
//}
//bool VoxelMap::triggerVoxelMapCollisionTestWithCollision(uint32_t nr_tests, VoxelMap* other)
//{
//  srand(time(0));
//  std::vector<Vector3f> this_testpoints(1);
//
//  for (uint32_t i = 0; i < nr_tests; i++)
//  {
//    this->clearMap();
//    other->clearMap();
//
//    float x, y, z;
//    x = (rand() / (double) RAND_MAX) * m_limits.x;
//    y = (rand() / (double) RAND_MAX) * m_limits.y;
//    z = (rand() / (double) RAND_MAX) * m_limits.z;
//    this_testpoints[0] = (Vector3f(x, y, z));
//
//    this->insertGlobalData(this_testpoints, eVT_OCCUPIED);
//    other->insertGlobalData(this_testpoints, eVT_OCCUPIED);
//    if (!this->collisionCheck(10, other, 10))
//    {/*if no collision has been found*/
//      return false;
//    }
//  }
//  return true;
//}
//bool VoxelMap::triggerVoxelMapCollisionTestNoCollision(uint32_t num_data_points, VoxelMap* other)
//{
//  std::vector<Vector3f> this_testpoints;
//  std::vector<Vector3f> other_testpoints;
//
//  // clear the map, insert non overlapping point pattern in both maps,
//  // check for NO collision
//  this->clearMap();
//  other->clearMap();
//  uint32_t num_points = 0;
//  for (uint32_t i = 0; i < (this->m_dim.x - 1) / 2; i++)
//  {
//    for (uint32_t j = 0; j < (this->m_dim.y - 1) / 2; j++)
//    {
//      for (uint32_t k = 0; k < (this->m_dim.z - 1) / 2; k++)
//      {
//        float x, y, z;
//        x = i * 2 * m_voxel_side_length + m_voxel_side_length / 2.0;
//        y = j * 2 * m_voxel_side_length + m_voxel_side_length / 2.0;
//        z = k * 2 * m_voxel_side_length + m_voxel_side_length / 2.0;
//        this_testpoints.push_back(Vector3f(x, y, z));
//        x = (i * 2 + 1) * m_voxel_side_length + m_voxel_side_length / 2.0;
//        y = (j * 2 + 1) * m_voxel_side_length + m_voxel_side_length / 2.0;
//        z = (k * 2 + 1) * m_voxel_side_length + m_voxel_side_length / 2.0;
//        other_testpoints.push_back(Vector3f(x, y, z));
//        if (num_points >= num_data_points)
//        {
//          goto OUT_OF_LOOP;
//        }
//        num_points++;
//      }
//    }
//  }
//  OUT_OF_LOOP:
//
//  this->insertGlobalData(this_testpoints, eVT_OCCUPIED);
//  other->insertGlobalData(other_testpoints, eVT_OCCUPIED);
//  // there shouldn't be a collision in this test!
//  return !this->collisionCheck(10, other, 10);
//
//}

} //end ns
} //end ns
