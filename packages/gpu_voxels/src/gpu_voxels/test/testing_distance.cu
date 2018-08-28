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
 * \author  Andreas Hermann <hermann@fzi.de>
 * \date    2016-12-23
 *
 */
//----------------------------------------------------------------------

#include <boost/test/unit_test.hpp>


#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>
#include <gpu_voxels/voxel/DistanceVoxel.hpp>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/test/testing_fixtures.hpp>

using namespace gpu_voxels;
using boost::shared_ptr;
typedef boost::shared_ptr<voxelmap::DistanceVoxelMap> DistMapSharedPtr;

BOOST_FIXTURE_TEST_SUITE(distance, ArgsFixture)

BOOST_AUTO_TEST_CASE(distance_correctness)
{
  float side_length = 1.f;

  // caution: shadows global variables defined in testing_fixtures
  //TODO  check for exact requirements of algorithms; assumptions appear to be:
  //        PBA: dimX==dimY and dimX % 64 == 0
  //        JFA: dimX and dimY must be even numbers
  //      suspected causes in PBA: 
  //        pba x/y in-place transpose operation assumes dimX == dimY
  //        pba x/y in-place transpose uses constant sized shared memory cache, PBA_TILE_DIM=16
  int dimX = 128;
  int dimY = dimX;
  int dimZ = 131;

  DistMapSharedPtr exact_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_DISTANCE_VOXELMAP));
  DistMapSharedPtr jfa_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_DISTANCE_VOXELMAP));
  DistMapSharedPtr pba_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_DISTANCE_VOXELMAP));

  // Create obstacle points: //TODO: Change this to random points
  std::vector<Vector3f> obstacles;
  geometry_generation::createEquidistantPointsInBox(numberOfPoints, Vector3ui(dimX, dimY, dimZ), side_length, obstacles);

  std::cout << "DEBUG distance test nrvox=" << numberOfPoints << ", dimX=" << dimX << ", dimY="<<dimY<<", dimZ="<<dimZ<<std::endl;

  PERF_MON_START("distance_correctness");
  for(int i = 0; i < iterationCount; i++)
  {
    exact_dist_map->clearMap();
    jfa_dist_map->clearMap();
    pba_dist_map->clearMap();

    exact_dist_map->insertPointCloud(obstacles, eBVM_OCCUPIED);
    jfa_dist_map->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pba_dist_map->insertPointCloud(obstacles, eBVM_OCCUPIED);

    // calculate distances using three different algorithms:

    //std::cout << "Calculating JFA..." << std::endl;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_START("distance_jfa_timer");
    jfa_dist_map->jumpFlood3D();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_PRINT_AND_RESET_INFO("distance_jfa_timer", "jumpFlood3D done");
    std::cout << "... JFA distance done" << std::endl;

    //std::cout << "Calculating PBA..." << std::endl;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_START("distance_pba_timer");
    pba_dist_map->parallelBanding3D(1, 1, 1);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_PRINT_AND_RESET_INFO("distance_pba_timer", "parallelBanding3D done");
    std::cout << "...PBA distance done" << std::endl;

    //std::cout << "Calculating exact distances..." << std::endl;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_START("distance_exact_timer");
    exact_dist_map->exactDistances3D(obstacles);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_PRINT_AND_RESET_INFO("distance_exact_timer", "exactDistances3D done");
    std::cout << "...naive exact distance done" << std::endl;

    DistanceVoxel::accumulated_diff diff_result;

    //std::cout << "Compare exact3DDistances and PBA 3D voxels:" << std::endl;
    diff_result = exact_dist_map->differences3D(pba_dist_map, false);
    BOOST_CHECK_MESSAGE((diff_result.maxerr < 0.01), "PBA failed!");
    if (!diff_result.maxerr < 0.01)
    {
      std::cout << diff_result.str() << std::endl;
    }

    //std::cout << "Compare exact3DDistances and JFA 3D voxels:" << std::endl;
    diff_result = exact_dist_map->differences3D(jfa_dist_map, false);
    BOOST_CHECK_MESSAGE((diff_result.maxerr < 0.09), "JFA failed!");
    if (!diff_result.maxerr < 0.09)
    {
      std::cout << diff_result.str() << std::endl;
    }

    //std::cout << "Compare PBA and JFA 3D voxels:" << std::endl;
    diff_result = jfa_dist_map->differences3D(pba_dist_map, false);
    BOOST_CHECK_MESSAGE((diff_result.maxerr < 0.09), "Sanity check failed!");
    if (!diff_result.maxerr < 0.09)
    {
      std::cout << diff_result.str() << std::endl;
    }
  }
}

BOOST_AUTO_TEST_CASE(distance_extraction)
{
  float side_length = 1.f;
  
  // caution: shadows global variables defined in testing_fixtures
  //TODO  check for exact requirements of algorithms; assumptions appear to be:
  //        PBA: dimX==dimY and dimX % 64 == 0
  //        JFA: dimX and dimY must be even numbers
  //      suspected causes in PBA: 
  //        pba x/y in-place transpose operation assumes dimX == dimY
  //        pba x/y in-place transpose uses constant sized shared memory cache, PBA_TILE_DIM=16
  int dimX = 64;
  int dimY = dimX;
  int dimZ = 64;

  DistMapSharedPtr pba_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_DISTANCE_VOXELMAP));

  std::cout << "DEBUG distance_extraction test dimX=" << dimX << ", dimY="<<dimY<<", dimZ="<<dimZ<<std::endl;

  // test gatherVoxelsByIndex, getSquaredDistances, getDistances
  PERF_MON_START("distance_extraction");
  for(int iter = 0; iter < iterationCount; iter++)
  {
    //get squared distance of each Position
    std::vector<uint> h_indices;
    h_indices.push_back(0);
    h_indices.push_back(10);
    h_indices.push_back(100);
    h_indices.push_back(1000);
    h_indices.push_back(10000);

    pba_dist_map->clearMap();
    std::vector<Vector3f> obstacles;
    for (size_t i = 0; i < h_indices.size(); i++) 
    {
        Vector3i coords = voxelmap::mapToVoxelsSigned(h_indices[i], pba_dist_map->getDimensions());
        Vector3f f_coords = coords / side_length;
        obstacles.push_back(f_coords);
    }    
    pba_dist_map->insertPointCloud(obstacles, eBVM_OCCUPIED);

    // use gatherVoxelsByIndex
    thrust::device_vector<uint> d_indices(h_indices);
    thrust::device_vector<DistanceVoxel> d_voxels(h_indices.size());
      
    pba_dist_map->gatherVoxelsByIndex(&(*d_indices.begin()), &(*d_indices.end()), d_voxels.data());
    
    // validate voxels
    for (size_t i = 0; i < d_voxels.size(); i++) 
    {
        DistanceVoxel dv = d_voxels[i];
        std::cout << "d_voxels["<<i<<"] = " << dv.getObstacle() << std::endl;
        
        Vector3ui coords = voxelmap::mapToVoxels(h_indices[i], pba_dist_map->getDimensions());
        BOOST_CHECK_MESSAGE((coords == dv.getObstacle()), "Sanity check failed! gatherVoxelsByIndex did not return the expected voxel contents");
    }
        
    // get squared distances
    std::vector<int> squared_distances(h_indices.size());
    pba_dist_map->getSquaredDistancesToHost(h_indices, squared_distances);
    
    // get distances
    std::vector<int> distances(h_indices.size());
    pba_dist_map->getDistancesToHost(h_indices, distances);
            
    std::vector<uint> h_indices_neighbors(h_indices);
    for (size_t i = 0; i < h_indices_neighbors.size(); i++) 
    {
        h_indices_neighbors[i]++;
    }
    
    //std::cout << "Calculating PBA..." << std::endl;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    pba_dist_map->parallelBanding3D(1, 1, 1);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    
    // get neighbor distances
    std::vector<int> distances_neighbors(h_indices_neighbors.size());
    pba_dist_map->getDistancesToHost(h_indices_neighbors, distances_neighbors);

    // perform sanity checks
    for (size_t i = 0; i < h_indices_neighbors.size(); i++) 
    {
        BOOST_CHECK_MESSAGE((distances[i] == 0), "Sanity check failed! distance at obstacle not zero");
        BOOST_CHECK_MESSAGE((squared_distances[i] == 0), "Sanity check failed! squared_distance at obstacle not zero");
        BOOST_CHECK_MESSAGE((distances[i] != distances_neighbors[i]), "Sanity check failed! distance equals neighbors distance");
//         std::cout << "distances["<<i<<"]=" << distances[i] << std::endl;
//         std::cout << "squared_distances["<<i<<"]=" << squared_distances[i] << std::endl;
//         std::cout << "distances_neighbors["<<i<<"]=" << distances_neighbors[i] << std::endl;
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()


