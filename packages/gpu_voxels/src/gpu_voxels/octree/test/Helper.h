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

#ifndef HELPER_H_
#define HELPER_H_

#include <gpu_voxels/octree/test/ArgumentHandling.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/octree/Voxel.h>

#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/BitVector.h>

namespace gpu_voxels {
namespace NTree {
namespace Test {

static const std::size_t RAND_SEED = 2746135025UL;

bool testAndInitDevice();

thrust::host_vector<gpu_voxels::Vector3ui> linearPoints(voxel_count num_points, OctreeVoxelID maxValue);

//struct Trafo_Point_to_Voxel
//{
//  __host__ __device__ Voxel operator()(const gpu_voxels::Vector3ui x)
//  {
//    return Voxel(morton_code60(x), x, MAX_OCCUPANCY);
//  }
//};

//void trafoPointToVoxel(thrust::host_vector<gpu_voxels::Vector3ui>& h_points,
//                       thrust::device_vector<Voxel>& d_voxel)
//{
//  thrust::transform(h_points.begin(), h_points.end(), d_voxel.begin(), Trafo_Point_to_Voxel());
//}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool buildOctree(NTree<branching_factor, level_count, InnerNode, LeafNode>* tree, std::vector<Vector3f>& points,
                 uint32_t& num_points,
                 BuildResult<branching_factor, level_count, InnerNode, LeafNode>& build_result,
                 float additional_scaling = 1.0f, bool free_bounding_box = false, Vector3i offset = Vector3i(0, 0, 0))
{
  std::size_t num_voxel = pow(branching_factor, level_count - 1);
  std::cout << gpu_voxels::getDeviceMemoryInfo();

  // thrust::device_std::std::vector<voxel_id> voxel;

  //  for (uint32_t i = 0; i < num_points; ++i)
  //    if (voxel[i] == 41)
  //      printf("VOXEL 41 FOUND\n");

  //  for (voxel_id i = 0; i < num_points; ++i)
  //    printf("Voxel %i: %lu\n", i, (int64_t) voxel[i]);

  printf("create octree....\n");

  build_result.center = gpu_voxels::Vector3ui(0, 0, 0);
  float scaling = 1000.0f / tree->m_resolution * additional_scaling;

  if (!points.empty())
  {
    std::vector<gpu_voxels::Vector3ui> pts;
    transformPointCloud(points, pts, build_result.map_dimensions, scaling);
    build_result.h_points = thrust::host_vector<gpu_voxels::Vector3ui>(pts.begin(), pts.end());
    num_points = pts.size();
  }
  else
  {
    build_result.h_points = linearPoints(num_points, num_voxel);
    build_result.map_dimensions.x = build_result.map_dimensions.y = build_result.map_dimensions.z =
        (uint32_t) ceil(pow(num_voxel, 1.0 / 3));
  }
  printf("num_points: %u\n", num_points);
  printf("MapDim x=%u y=%u z=%u\n", build_result.map_dimensions.x, build_result.map_dimensions.y,
         build_result.map_dimensions.z);

  OctreeVoxelID max_morton_code = morton_code60(build_result.map_dimensions.x - 1,
                                          build_result.map_dimensions.y - 1,
                                          build_result.map_dimensions.z - 1);
  OctreeVoxelID tree_limit = (uint64_t) pow(branching_factor, level_count - 1) - 1;
  printf("Max morton code: %lu Map morton limit: %lu\n", max_morton_code, tree_limit);
  if (max_morton_code > tree_limit)
  {
    printf("NTree is too small for data!\n");
    return false;
  }

  timespec time1 = getCPUTime();
  build_result.center = tree->m_center - (build_result.map_dimensions / gpu_voxels::Vector3ui(2));

  // shift to center of octree
  for (uint32_t i = 0; i < num_points; ++i)
  {
    build_result.h_points[i] = build_result.h_points[i] + build_result.center;
    build_result.h_points[i].x = build_result.h_points[i].x + offset.x;
    build_result.h_points[i].y = build_result.h_points[i].y + offset.y;
    build_result.h_points[i].z = build_result.h_points[i].z + offset.z;
  }

//  thrust::device_vector<Voxel> d_voxel(build_result.h_points.size());
//  thrust::transform(build_result.h_points.begin(), build_result.h_points.end(), d_voxel.begin(),
//                    Trafo_Point_to_Voxel());

  tree->build(build_result.h_points, free_bounding_box);

  build_result.time = timeDiff(time1, getCPUTime());
  printf("build: %f ms\n", build_result.time);

  build_result.o = tree;
  build_result.mem_usage = build_result.o->getMemUsage();
  build_result.octree_leaf_nodes = build_result.o->allocLeafNodes;
  build_result.octree_inner_nodes = build_result.o->allocInnerNodes;

  //tree->print2();

  std::cout << gpu_voxels::getDeviceMemoryInfo();

  printf("AllocOnnerNodes: %u    AllocLeafNodes: %u\n", build_result.o->allocInnerNodes,
         build_result.o->allocLeafNodes);

//std::cout << "VOXEL CHECK " << voxel[12] << std::endl;

  printf("octree created\n");

//  for (voxel_id i = 0; i < num_points; ++i)
//    printf("Voxel %i: %lu\n", i, (int64_t) hVoxel[i]);

//o->print();

  return true;
}

template<int VTF_SIZE>
thrust::host_vector<BitVector<VTF_SIZE> > randomVoxelTypes(voxel_count num_points)
{
  thrust::host_vector<BitVector<VTF_SIZE> > rand_types(num_points);

  for (voxel_count i = 0; i < num_points; ++i)
  {
    // set 3 random bits per Vector
    rand_types[i].setBit(rand() % VTF_SIZE);
    rand_types[i].setBit(rand() % VTF_SIZE);
    rand_types[i].setBit(rand() % VTF_SIZE);
  }
  return rand_types;
}

void getRandomPlan(std::vector<Vector3f>& robot, std::vector<Vector3f>& random_plan, int path_points,
                   Vector3f map_size, float step_size = 0.1f);

void getRandomPlans(std::vector<Provider::Provider_Parameter>& parameter);

}

}
}

#endif /* HELPER_H_ */
