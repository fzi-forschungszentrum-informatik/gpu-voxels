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
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/test/testing_fixtures.hpp>

using namespace gpu_voxels;
using boost::shared_ptr;
typedef boost::shared_ptr<voxelmap::DistanceVoxelMap> DistMapSharedPtr;

BOOST_FIXTURE_TEST_SUITE(distance, ArgsFixture)

BOOST_AUTO_TEST_CASE(distance_correctness)
{
  float side_length = 1.f;

  DistMapSharedPtr exact_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(128, 128, 128), side_length, MT_DISTANCE_VOXELMAP));
  DistMapSharedPtr jfa_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(128, 128, 128), side_length, MT_DISTANCE_VOXELMAP));
  DistMapSharedPtr pba_dist_map = DistMapSharedPtr(new voxelmap::DistanceVoxelMap(Vector3ui(128, 128, 128), side_length, MT_DISTANCE_VOXELMAP));

  // Create obstacle points: //TODO: Change this to random points
  std::vector<Vector3f> obstacles;
  geometry_generation::createEquidistantPointsInBox(numberOfPoints, Vector3ui(128, 128, 128), side_length, obstacles);

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
    jfa_dist_map->jumpFlood3D(cMAX_THREADS_PER_BLOCK, 1, false);
    std::cout << "...Done" << std::endl;
    //std::cout << "Calculating PBA..." << std::endl;
    pba_dist_map->parallelBanding3D(1, 1, 4, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 0);
    std::cout << "...Done" << std::endl;
    //std::cout << "Calculating exact distances..." << std::endl;
    exact_dist_map->exactDistances3D(obstacles);

    DistanceVoxel::accumulated_diff diff_result;

    //std::cout << "Compare exact3DDistances and PBA 3D voxels:" << std::endl;
    diff_result = exact_dist_map->differences3D(pba_dist_map, false);
    BOOST_CHECK_MESSAGE((diff_result.maxerr < 0.01), "PBA failed!");

    //std::cout << "Compare exact3DDistances and JFA 3D voxels:" << std::endl;
    diff_result = exact_dist_map->differences3D(jfa_dist_map, false);
    BOOST_CHECK_MESSAGE((diff_result.maxerr < 0.09), "JFA failed!");

    //std::cout << "Compare PBA and JFA 3D voxels:" << std::endl;
    diff_result = jfa_dist_map->differences3D(pba_dist_map, false);
    BOOST_CHECK_MESSAGE((diff_result.maxerr < 0.09), "Sanity check failed!");
  }
}



BOOST_AUTO_TEST_SUITE_END()


