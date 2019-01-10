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
 * \author  Matthias Wagner <mwagner@fzi.de>
 * \date    2014-06-13
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/voxelmap/Tests.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/voxel/SVCollider.hpp>
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/helpers/GeometryGeneration.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/test/testing_fixtures.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/test/unit_test.hpp>
#include "icl_core_performance_monitor/PerformanceMonitor.h"

using namespace gpu_voxels;
using namespace voxelmap;
using namespace geometry_generation;

BOOST_FIXTURE_TEST_SUITE(voxelmap, ArgsFixture)

BOOST_AUTO_TEST_CASE(addresScheme)
{
  PERF_MON_START("addresScheme");
  for(int i = 0; i < iterationCount; i++)
  {
    size_t nr_of_tests = 30000;
    float voxel_side_length = 1.f;
    bool success = true;

    test::triggerAddressingTest<BitVectorVoxel>(Vector3ui(dimX, dimY, dimZ), voxel_side_length, nr_of_tests, &success);

    BOOST_CHECK_MESSAGE(success, "FloatCoords to Voxel == Voxel to FloatCoords");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("addresScheme", "addresScheme", "voxelmap");
  }
}


BOOST_AUTO_TEST_CASE(offset)
{
  PERF_MON_START("offset");
  for(int i = 0; i < iterationCount; i++)
  {
  // (offset.z * (int32_t)dimensions.x * (int32_t)dimensions.y + offset.y * (int32_t)dimensions.x + offset.x);

    Vector3ui sdim(15, 16, 17);
    Vector3i  offset(0, -2, 0);
    ptrdiff_t ptr_offset = getVoxelIndexSigned(sdim, offset);
    ptrdiff_t ground_truth = -30;
    BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Negative Offset -30 correct.");

    offset = Vector3i(1, 1, 1);
    ptr_offset = getVoxelIndexSigned(sdim, offset);
    ground_truth = 256;
    BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Positive Offset 256 correct.");

    offset = Vector3i(1, -2, -3);
    ptr_offset = getVoxelIndexSigned(sdim, offset);
    ground_truth = -749;
    BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Negative Offset -749 correct.");

    offset = Vector3i(3, -2, -1);
    ptr_offset = getVoxelIndexSigned(sdim, offset);
    ground_truth = -267;
    BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Negative Offset -267 correct.");

    offset = Vector3i(-1, -2, 3);
    ptr_offset = getVoxelIndexSigned(sdim, offset);
    ground_truth = 689;
    BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Positive Offset 689 correct.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("offset", "offset", "voxelmap");
  }
}


BOOST_AUTO_TEST_CASE(collision)
{
  PERF_MON_START("collision");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;
    ProbVoxelMap map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);
    ProbVoxelMap map_2(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);

    std::vector<Vector3f> this_testpoints;

    createEquidistantPointsInBox(numberOfPoints, Vector3ui(dimX, dimY, dimZ), side_length, this_testpoints);

    map_1.insertPointCloud(this_testpoints, eBVM_OCCUPIED);
    map_2.insertPointCloud(this_testpoints, eBVM_OCCUPIED);

    BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1) == (size_t)numberOfPoints, "All collisions detected.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collision", "collision", "voxelmap");
  }
}

//! Create two 3x3x3 boxes that overlap by 8 voxels.
//! Then collide with and without offset.
BOOST_AUTO_TEST_CASE(collision_with_offset)
{
  PERF_MON_START("collision_with_offset");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;
    ProbVoxelMap map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);
    ProbVoxelMap map_2(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);

    std::vector<Vector3f> this_testpoints1;
    std::vector<Vector3f> this_testpoints2;

    this_testpoints1 = createBoxOfPoints( Vector3f(2.1, 2.1, 2.1), Vector3f(4.1, 4.1, 4.1), 0.5);
    this_testpoints2 = createBoxOfPoints( Vector3f(3.1, 3.1, 3.1), Vector3f(5.1, 5.1, 5.1), 0.5);

    map_1.insertPointCloud(this_testpoints1, eBVM_OCCUPIED);
    map_2.insertPointCloud(this_testpoints2, eBVM_OCCUPIED);

    BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1) == 8, "All collisions without offset detected.");
    BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1, Vector3i(-1,0,-1)) == 18, "All collisions with negative offset detected.");
    BOOST_CHECK_MESSAGE(map_2.collideWith(&map_1, 0.1, Vector3i(1,0,1)) == 18, "All collisions with positive offset detected.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collision_with_offset", "collision_with_offset", "voxelmap");
  }
}

BOOST_AUTO_TEST_CASE(no_collision)
{
  PERF_MON_START("no_collision");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;

    std::vector<Vector3f> this_testpoints;
    std::vector<Vector3f> other_testpoints;

    ProbVoxelMap map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);
    ProbVoxelMap map_2(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);

    // clear the map, insert non overlapping point pattern in both maps,
    // check for NO collision
    createNonOverlapping3dCheckerboard(numberOfPoints, Vector3ui(dimX, dimY, dimZ), side_length, this_testpoints, other_testpoints);

    map_1.insertPointCloud(this_testpoints, eBVM_OCCUPIED);
    map_2.insertPointCloud(other_testpoints, eBVM_OCCUPIED);

    // there shouldn't be a collision in this test!
    BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1) == 0, "No collisions detected.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("no_collision", "no_collision", "voxelmap");
  }
}

/**
 * @brief Test for windowed collsion checks.
 *
 */
BOOST_AUTO_TEST_CASE(collision_bitshift)
{
  PERF_MON_START("collision_bitshift");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;
    BitVectorVoxelMap map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP);
    BitVectorVoxelMap map_2(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP);

    std::vector<Vector3f> points;
    points.push_back(Vector3f(2,2,2));
    points.push_back(Vector3f(2,2,3));
    points.push_back(Vector3f(2,3,2));
    points.push_back(Vector3f(2,3,3));
    points.push_back(Vector3f(3,2,2));
    points.push_back(Vector3f(3,2,3));
    points.push_back(Vector3f(3,3,2));
    points.push_back(Vector3f(3,3,3));

    const BitVoxelMeaning type_1 = BitVoxelMeaning(eBVM_SWEPT_VOLUME_START+40);

    map_1.insertPointCloud(points, type_1);

    size_t window_size = 20;
    SVCollider collider(1.0, window_size);
    size_t num_collisions = 0;

    BitVectorVoxel types_voxel;
    /* The basic idea is as follows:
   * map_1 contains voxels with BVM type_1. Then in map_2 the same voxels are inserted, but with
   * another BVM.
   * These voxels are checked for collision with map_1 with a given window size. This is repeated
   * for different BVMs for map_2. Collisions should only occur, when the BVMs are close enough
   * for the windowed collider. Otherwise there should be no collisions present.
   */
    for (size_t i = type_1 - window_size-9; i <= type_1 + window_size+9; ++i)
    {
      map_2.clearMap();
      map_2.insertPointCloud(points, BitVoxelMeaning(i));
      num_collisions = map_1.collisionCheckBitvector(&map_2, collider, types_voxel.bitVector());
      //    std::cout << "Num collisions with diff " << static_cast<int32_t>(i - type_1) << ": " << num_collisions << std::endl;

      // Make sure all collisions accur with points of type type_1
      if (num_collisions > 0)
      {
        types_voxel.bitVector().clearBit(type_1);
        BOOST_CHECK(types_voxel.bitVector().isZero());
      }

      // Collisions should only occur when the window and type_1 are overlapping.
      if (i < type_1 - window_size || i > type_1 + window_size)
      {
        BOOST_CHECK(num_collisions == 0);
      }
      else
      {
        BOOST_CHECK(num_collisions == points.size());
      }
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collision_bitshift", "collision_bitshift", "voxelmap");
  }
}

/**
 * @brief This test is for the bitshifting only. It basically sets the same bit in two maps
 * once by bitshifting and once by explicit set and then checks, wheter both maps collide and
 * only collide in these shifted/set voxels.
 *
 * Please note that the bitshift collision is tested in another boost-test.
 */
BOOST_AUTO_TEST_CASE(bitvoxelmap_bitshift)
{
  PERF_MON_START("bitvoxelmap_bitshift");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;
    BitVectorVoxelMap map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP);
    BitVectorVoxelMap map_2(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP);

    // add some arbitrary voxels
    std::vector<Vector3f> points;
    points.push_back(Vector3f(0.3,0.3,0.3));
    points.push_back(Vector3f(1.3,1.3,1.3));
    points.push_back(Vector3f(2.3,2.3,2.3));
    points.push_back(Vector3f(3.3,3.3,3.3));

    uint32_t shift_size = 0;
    const uint32_t shift_start = 50;

    // That's our starting point. In the shifted map voxels get always inserted with this BVM
    const uint32_t type_int = eBVM_SWEPT_VOLUME_START + shift_start;
    const BitVoxelMeaning type_2 = BitVoxelMeaning(type_int);

    /* The key idea of the following is:
   * insert the same points with BVM type_int and type_int-shift_size in different maps.
   * Then shift the BVMs of map_2 such that in both maps the points should have the same BVM.
   * Then, all voxels (and only those voxels) should be colliding.
   */
    while (shift_size < shift_start + eBVM_SWEPT_VOLUME_START - 5)
    {
      map_1.clearMap();
      map_2.clearMap();

      // Insert voxels in map_2 with fix BVM
      map_2.insertPointCloud(points, type_2);

      // Insert voxels in map_1 with a BVM that is shift_size "away" from type_2
      const BitVoxelMeaning type_1 = BitVoxelMeaning(type_int - shift_size);
      map_1.insertPointCloud(points, type_1);

      // Now shift voxels in map_2, so they should have the same BVM as those in map_1
      map_2.shiftLeftSweptVolumeIDs(shift_size);

      size_t num_collisions = 0;
      BitVectorVoxel types_voxel;
      size_t window_size = 1;
      SVCollider collider(1.0, window_size);
      num_collisions = map_1.collisionCheckBitvector(&map_2, collider, types_voxel.bitVector());

      if (shift_size <= shift_start)
      {
        BOOST_CHECK(num_collisions == points.size());
      }
      else
      {
        BOOST_CHECK(num_collisions == 0);
      }
      shift_size++;
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxelmap_bitshift", "bitvoxelmap_bitshift", "voxelmap");
  }
}



/**
 * @brief Test for collision types
 *
 */
BOOST_AUTO_TEST_CASE(collision_with_types_probab)
{
  PERF_MON_START("collision_with_types_probab");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;
    BitVectorVoxelMap map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP);
    ProbVoxelMap map_2(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP);

    std::vector<Vector3f> points;
    points.push_back(Vector3f(2,2,2));
    points.push_back(Vector3f(2,2,3));
    points.push_back(Vector3f(2,3,2));
    points.push_back(Vector3f(2,3,3));
    points.push_back(Vector3f(3,2,2));
    points.push_back(Vector3f(3,2,3));
    points.push_back(Vector3f(3,3,2));
    points.push_back(Vector3f(3,3,3));

    map_1.insertPointCloud(points, BitVoxelMeaning(eBVM_SWEPT_VOLUME_START+10));
    map_1.insertPointCloud(points, BitVoxelMeaning(eBVM_SWEPT_VOLUME_START+42));
    map_1.insertPointCloud(points, BitVoxelMeaning(eBVM_SWEPT_VOLUME_START+99));

    map_2.insertPointCloud(points, eBVM_OCCUPIED);

    size_t num_collisions = 0;

    BitVectorVoxel coll_types;
    num_collisions = map_1.collideWithTypes(&map_2, coll_types, ProbabilisticVoxel::probabilityToFloat(cSENSOR_MODEL_OCCUPIED));

    std::cout << "Num colls = " << num_collisions << std::endl;
    std::cout << "Col types are " << coll_types << std::endl;


    BOOST_CHECK(num_collisions == points.size());
    BOOST_CHECK(coll_types.bitVector().getBit(eBVM_SWEPT_VOLUME_START+10));
    BOOST_CHECK(coll_types.bitVector().getBit(eBVM_SWEPT_VOLUME_START+42));
    BOOST_CHECK(coll_types.bitVector().getBit(eBVM_SWEPT_VOLUME_START+99));

    coll_types.bitVector().clearBit(eBVM_SWEPT_VOLUME_START+10);
    coll_types.bitVector().clearBit(eBVM_SWEPT_VOLUME_START+42);
    coll_types.bitVector().clearBit(eBVM_SWEPT_VOLUME_START+99);

    BOOST_CHECK(coll_types.bitVector().noneButEmpty());

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collision_with_types_probab", "collision_with_types_probab", "voxelmap");
  }
}

/**
 * @brief Test for selfcollisions in metapointclouds
 *
 */
BOOST_AUTO_TEST_CASE(selfcollision_metapointcloud)
{
  PERF_MON_START("selfcollision_metapointcloud");
  for(int i = 0; i < iterationCount; i++)
  {
    // force kernel calls that run in a single block as well as multiple blocks
    for(int j = 2; j < 11; j+=8 )
    {
      float side_length = 1.f;
      BitVectorVoxelMap map(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP);


      // PART 1: No optional parameters are given. Default meanings are used:


      std::vector< std::vector<Vector3f> > meta_vectors_colliding;

      // First three boxex overlap, last one is seperated
      meta_vectors_colliding.push_back(createBoxOfPoints( Vector3f(2.1, 2.1, 2.1), Vector3f(5.1, 5.1, 5.1), 1.0/j));
      meta_vectors_colliding.push_back(createBoxOfPoints( Vector3f(4.1, 4.1, 4.1), Vector3f(7.1, 7.1, 7.1), 1.0/j));
      meta_vectors_colliding.push_back(createBoxOfPoints( Vector3f(6.1, 6.1, 6.1), Vector3f(9.1, 9.1, 9.1), 1.0/j));
      meta_vectors_colliding.push_back(createBoxOfPoints( Vector3f(10.1, 10.1, 10.1), Vector3f(12.1, 12.1, 12.1), 1.0/j));

      MetaPointCloud meta_pc_colliding_host(meta_vectors_colliding);

      bool self_coll = map.insertMetaPointCloudWithSelfCollisionCheck(&meta_pc_colliding_host);
      BOOST_CHECK(self_coll == true);



      std::vector< std::vector<Vector3f> > meta_vectors_noncolliding;

      // All boxes are seperated and do not collide with previous ones:
      meta_vectors_noncolliding.push_back(createBoxOfPoints( Vector3f(30.1, 30.1, 30.1), Vector3f(32.1, 32.1, 32.1), 1.0/j));
      meta_vectors_noncolliding.push_back(createBoxOfPoints( Vector3f(40.1, 40.1, 40.1), Vector3f(43.1, 43.1, 43.1), 1.0/j));
      meta_vectors_noncolliding.push_back(createBoxOfPoints( Vector3f(60.1, 60.1, 60.1), Vector3f(64.1, 64.1, 64.1), 1.0/j));
      meta_vectors_noncolliding.push_back(createBoxOfPoints( Vector3f(65.1, 65.1, 65.1), Vector3f(68.1, 68.1, 68.1), 1.0/j));


      MetaPointCloud meta_pc_noncolliding_host(meta_vectors_noncolliding);

      self_coll = map.insertMetaPointCloudWithSelfCollisionCheck(&meta_pc_noncolliding_host);
      BOOST_CHECK(self_coll == false);

      map.clearMap();

      // PART 2: All params are given:

      std::vector<BitVoxelMeaning> voxel_meanings(4);

      // arbitrary offset for demonstration purposes. Usefull if you need more than one robot.
      voxel_meanings[0] = BitVoxelMeaning(50 + 1);
      voxel_meanings[1] = BitVoxelMeaning(50 + 2);
      voxel_meanings[2] = BitVoxelMeaning(50 + 3);
      voxel_meanings[3] = BitVoxelMeaning(50 + 4);

      std::vector<BitVector<BIT_VECTOR_LENGTH> > collision_masks(4);
      BitVector<BIT_VECTOR_LENGTH> tmp_coll_mask; // All collisions disabled
      tmp_coll_mask = ~tmp_coll_mask;
      tmp_coll_mask.clearBit(eBVM_COLLISION); // mask out the colliding bit

      // check the assertions on size equality of all input paras!! Seems not to be checked!

      collision_masks[0] = tmp_coll_mask; // set all to True
      collision_masks[1] = tmp_coll_mask; // set all to True
      collision_masks[2] = tmp_coll_mask; // set all to True
      collision_masks[3] = tmp_coll_mask; // set all to True

      collision_masks[0].clearBit(50 + 2); // Disbale collision of first cloud with second cloud ==> No collisions should be detected
      collision_masks[1].clearBit(50 + 1); // Disbale collision of second cloud with first cloud ==> Only collisions with third cloud should be detected

      collision_masks[0].clearBit(50 + 1); // reset own meaning, otherwise we would see a selfcollision if more than one point falls into the same voxel
      collision_masks[1].clearBit(50 + 2);
      collision_masks[2].clearBit(50 + 3);
      collision_masks[3].clearBit(50 + 4);


      BitVector<BIT_VECTOR_LENGTH> colliding_meanings; // return value

      self_coll = map.insertMetaPointCloudWithSelfCollisionCheck(&meta_pc_colliding_host, voxel_meanings, collision_masks, &colliding_meanings);

      //std::cout << "colliding_meanings 1:" << colliding_meanings << std::endl;

      BOOST_CHECK(self_coll == true);
      BOOST_CHECK(colliding_meanings.getBit(50 + 1) == false);
      BOOST_CHECK(colliding_meanings.getBit(50 + 2) == true);
      BOOST_CHECK(colliding_meanings.getBit(50 + 3) == true);
      BOOST_CHECK(colliding_meanings.getBit(50 + 4) == false);

      colliding_meanings.clearBit(50 + 2); // reset the expected collisions
      colliding_meanings.clearBit(50 + 3);
      BOOST_CHECK(colliding_meanings.isZero() == true); // everything else should be zero

      // now insert the noncolliding parts:

      collision_masks[0] = tmp_coll_mask; // activate all collisions
      collision_masks[1] = tmp_coll_mask;
      collision_masks[2] = tmp_coll_mask;
      collision_masks[3] = tmp_coll_mask;

      collision_masks[0].clearBit(50 + 1); // reset own mask
      collision_masks[1].clearBit(50 + 2);
      collision_masks[2].clearBit(50 + 3);
      collision_masks[3].clearBit(50 + 4);

      colliding_meanings.clear();
      self_coll = map.insertMetaPointCloudWithSelfCollisionCheck(&meta_pc_noncolliding_host, voxel_meanings, collision_masks, &colliding_meanings);
      //std::cout << "colliding_meanings 2:" << colliding_meanings << std::endl;
      BOOST_CHECK(self_coll == false);
      BOOST_CHECK(colliding_meanings.isZero() == true);

      PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("selfcollision_metapointcloud", "selfcollision_metapointcloud", "voxelmap");
    }
  }
}

BOOST_AUTO_TEST_CASE(iostream_bitvoxel)
{
  PERF_MON_START("iostream_bitvoxel");
  for(int i = 0; i < iterationCount; i++)
  {
    BitVectorVoxel my_voxel;

    my_voxel.insert(eBVM_FREE);
    my_voxel.insert(eBVM_COLLISION);
    my_voxel.insert(eBVM_UNKNOWN);
    my_voxel.insert(eBVM_SWEPT_VOLUME_START);
    my_voxel.insert(BitVoxelMeaning(111));
    my_voxel.insert(eBVM_SWEPT_VOLUME_END);
    my_voxel.insert(eBVM_UNDEFINED);

    std::string ground_truth = "1011100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011";

    std::stringstream ss;
    ss << my_voxel;

    BOOST_CHECK_MESSAGE(ss.str() == ground_truth, "Outstream");


    BitVectorVoxel my_voxel2;
    ss.str(ground_truth);
    ss >> my_voxel2;

    ss.str(std::string());
    ss << my_voxel2;


    BOOST_CHECK_MESSAGE(ss.str() == ground_truth, "Instream");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("iostream_bitvoxel", "iostream_bitvoxel", "voxelmap");
  }
}

BOOST_AUTO_TEST_CASE(voxelmap_closing)
{
  PERF_MON_START("voxelmap_closing");
  float side_length = 0.1f;
  Vector3ui dim(16, 16, 16);
  float occupied_threshold = 0.0f;
  ProbabilisticVoxel voxelmap_h[dim.x * dim.y * dim.z];

  for(int i = 0; i < iterationCount; i++)
  {
    Vector3f center_min = Vector3f(0.9,0.9,0.9);
    Vector3f center_max = Vector3f(1.1,1.1,1.1);
    std::vector<Vector3ui> box_coordinates = geometry_generation::createBoxOfPoints(center_min, center_max, side_length, side_length);

    // Check normal insertion
    {
      ProbVoxelMap voxelmap_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap_d.insertCoordinateList(box_coordinates, eBVM_OCCUPIED);

      HANDLE_CUDA_ERROR(cudaMemcpy((void**) &voxelmap_h, voxelmap_d.getDeviceDataPtr(), sizeof(ProbabilisticVoxel) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
      for (size_t x = 0; x < dim.x; x++)
      {
        for (size_t y = 0; y < dim.y ; y++)
        {
          for (size_t z = 0; z < dim.z; z++)
          {
            size_t index = getVoxelIndexUnsigned(dim, Vector3ui(x, y, z));
            bool occupied = voxelmap_h[index].isOccupied(occupied_threshold);
            BOOST_CHECK_MESSAGE(occupied == (9 <= x && x <= 11 && 9 <= y && y <= 11 && 9 <= z && z <= 11), "Voxel at " << x << ", " << y << ", " << z << " has wrong occupation=" << occupied);
          }
        }
      }
    }

    // Check dilated insertion
    {
      ProbVoxelMap voxelmap_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap_d.insertDilatedCoordinateList(box_coordinates, eBVM_OCCUPIED);

      HANDLE_CUDA_ERROR(cudaMemcpy((void**) &voxelmap_h, voxelmap_d.getDeviceDataPtr(), sizeof(ProbabilisticVoxel) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
      for (size_t x = 0; x < dim.x; x++)
      {
        for (size_t y = 0; y < dim.y ; y++)
        {
          for (size_t z = 0; z < dim.z; z++)
          {
            size_t index = getVoxelIndexUnsigned(dim, Vector3ui(x, y, z));
            bool occupied = voxelmap_h[index].isOccupied(occupied_threshold);
            BOOST_CHECK_MESSAGE(occupied == (8 <= x && x <= 12 && 8 <= y && y <= 12 && 8 <= z && z <= 12), "Voxel at " << x << ", " << y << ", " << z << " has wrong occupation=" << occupied);
          }
        }
      }
    }

    // Check closed insertion (dilation + erosion)
    {
      ProbVoxelMap voxelmap_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap_d.insertClosedCoordinateList(box_coordinates, eBVM_OCCUPIED, 1.0);

      HANDLE_CUDA_ERROR(cudaMemcpy((void**) &voxelmap_h, voxelmap_d.getDeviceDataPtr(), sizeof(ProbabilisticVoxel) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
      for (size_t x = 0; x < dim.x; x++)
      {
        for (size_t y = 0; y < dim.y ; y++)
        {
          for (size_t z = 0; z < dim.z; z++)
          {
            size_t index = getVoxelIndexUnsigned(dim, Vector3ui(x, y, z));
            bool occupied = voxelmap_h[index].isOccupied(occupied_threshold);
            BOOST_CHECK_MESSAGE(occupied == (9 <= x && x <= 11 && 9 <= y && y <= 11 && 9 <= z && z <= 11), "Voxel at " << x << ", " << y << ", " << z << " has wrong occupation=" << occupied);
          }
        }
      }
    }

    center_min = Vector3f(0.6,0.6,0.6);
    center_max = Vector3f(1.2,1.2,1.2);
    box_coordinates = geometry_generation::createBoxOfPoints(center_min, center_max, 2 * side_length, side_length);

    // Check lonely erosion
    {
      ProbVoxelMap voxelmap_src_d(dim, side_length, MT_PROBAB_VOXELMAP);
      ProbVoxelMap voxelmap_dest_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap_src_d.insertCoordinateList(box_coordinates, eBVM_OCCUPIED);
      voxelmap_src_d.erodeLonelyInto(voxelmap_dest_d);

      HANDLE_CUDA_ERROR(cudaMemcpy((void**) &voxelmap_h, voxelmap_dest_d.getDeviceDataPtr(), sizeof(ProbabilisticVoxel) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
      for (size_t x = 0; x < dim.x; x++)
      {
        for (size_t y = 0; y < dim.y ; y++)
        {
          for (size_t z = 0; z < dim.z; z++)
          {
            size_t index = getVoxelIndexUnsigned(dim, Vector3ui(x, y, z));
            bool occupied = voxelmap_h[index].isOccupied(occupied_threshold);
            BOOST_CHECK_MESSAGE(occupied == false, "Voxel at " << x << ", " << y << ", " << z << " has wrong occupation=" << occupied);
          }
        }
      }
    }

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxelmap_closing", "voxelmap_closing", "voxelmap");
  }
}

BOOST_AUTO_TEST_CASE(voxelmap_cloning)
{
  PERF_MON_START("voxelmap_cloning");
  float side_length = 0.1f;
  Vector3ui dim(16, 16, 16);
  float occupied_threshold = 0.0f;
  ProbabilisticVoxel voxelmap_h[dim.x * dim.y * dim.z];

  for(int i = 0; i < iterationCount; i++)
  {
    Vector3f center_min = Vector3f(0.9,0.9,0.9);
    Vector3f center_max = Vector3f(1.1,1.1,1.1);
    std::vector<Vector3ui> box_coordinates = geometry_generation::createBoxOfPoints(center_min, center_max, side_length, side_length);

    // Check cloning with correct dimensions
    {
      ProbVoxelMap voxelmap1_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap1_d.insertCoordinateList(box_coordinates, eBVM_OCCUPIED);

      ProbVoxelMap voxelmap2_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap2_d.clone(voxelmap1_d);

      HANDLE_CUDA_ERROR(cudaMemcpy((void**) &voxelmap_h, voxelmap2_d.getDeviceDataPtr(), sizeof(ProbabilisticVoxel) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
      for (size_t x = 0; x < dim.x; x++)
      {
        for (size_t y = 0; y < dim.y ; y++)
        {
          for (size_t z = 0; z < dim.z; z++)
          {
            size_t index = getVoxelIndexUnsigned(dim, Vector3ui(x, y, z));
            bool occupied = voxelmap_h[index].isOccupied(occupied_threshold);
            BOOST_CHECK_MESSAGE(occupied == (9 <= x && x <= 11 && 9 <= y && y <= 11 && 9 <= z && z <= 11), "Voxel at " << x << ", " << y << ", " << z << " has wrong occupation=" << occupied);
          }
        }
      }
    }

    // Check cloning with incorrect dimensions
    {
      ProbVoxelMap voxelmap1_d(dim, side_length, MT_PROBAB_VOXELMAP);
      voxelmap1_d.insertCoordinateList(box_coordinates, eBVM_OCCUPIED);

      ProbVoxelMap voxelmap2_d(dim, 2 * side_length, MT_PROBAB_VOXELMAP);
      voxelmap2_d.clone(voxelmap1_d);

      HANDLE_CUDA_ERROR(cudaMemcpy((void**) &voxelmap_h, voxelmap2_d.getDeviceDataPtr(), sizeof(ProbabilisticVoxel) * dim.x * dim.y * dim.z, cudaMemcpyDeviceToHost));
      for (size_t x = 0; x < dim.x; x++)
      {
        for (size_t y = 0; y < dim.y ; y++)
        {
          for (size_t z = 0; z < dim.z; z++)
          {
            size_t index = getVoxelIndexUnsigned(dim, Vector3ui(x, y, z));
            bool occupied = voxelmap_h[index].isOccupied(occupied_threshold);
            BOOST_CHECK_MESSAGE(!occupied, "Voxel at " << x << ", " << y << ", " << z << " has wrong occupation=" << occupied);
          }
        }
      }
    }

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxelmap_cloning", "voxelmap_cloning", "voxelmap");
  }
}

BOOST_AUTO_TEST_SUITE_END()



