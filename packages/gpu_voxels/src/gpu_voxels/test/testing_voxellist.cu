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
 * \date    2015-09-22
 *
 */
//----------------------------------------------------------------------
#include <sstream>
#include <gpu_voxels/voxellist/BitVoxelList.h>
#include <gpu_voxels/voxellist/CountingVoxelList.h>
#include <gpu_voxels/voxelmap/ProbVoxelMap.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>
#include <gpu_voxels/test/testing_fixtures.hpp>

#include <boost/make_shared.hpp>

#include <boost/test/unit_test.hpp>

using namespace gpu_voxels;
using namespace voxellist;
using namespace voxelmap;
using namespace geometry_generation;


BOOST_FIXTURE_TEST_SUITE(voxellists, ArgsFixture)


BOOST_AUTO_TEST_CASE(collide_bitvoxellist_with_countingpermeaning_bitvoxellist)
{
  PERF_MON_START("collide_bitvoxellist_with_countingpermeaning_bitvoxellist");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.0f;

    BitVectorVoxelList* obstacle = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);
    BitVectorVoxelList* sweptVolume = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);

    Vector3f b_min(8.0, 8.0, 8.0);
    Vector3f b_max(11.0, 11.0, 11.0);

    //create the obstacle and load it in a map
    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;
    box_clouds.push_back(createBoxOfPoints(b_min, b_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();
    obstacle->insertMetaPointCloud(boxes, eBVM_OCCUPIED);

    //generate the swept volume
    std::vector<std::vector<Vector3f> > sweptVolumeCloud;
    std::vector<BitVoxelMeaning> sweptVolumeMeanings;
    const float swept_ratio_delta = 0.02;
    const int num_swept_volumes = 50;
    const int min = 0;
    const int max = 30;
    for (int i = 0; i < num_swept_volumes; i++)
    {
      Vector3f position((max-min) * swept_ratio_delta * i + min, (max-min) * swept_ratio_delta * i + min, (max-min) * swept_ratio_delta * i + min);
      Vector3f positionM(position.x + 2.5f, position.y + 2.5f, position.z + 2.5f);
      sweptVolumeMeanings.push_back(BitVoxelMeaning(eBVM_SWEPT_VOLUME_START + i));
      sweptVolumeCloud.push_back(createBoxOfPoints(position, positionM, delta));
    }

    MetaPointCloud sweptVolumeMeta(sweptVolumeCloud);
    sweptVolumeMeta.syncToDevice();
    sweptVolume->insertMetaPointCloud(sweptVolumeMeta, sweptVolumeMeanings);

    //collide the two lists
    std::vector<size_t> collisions_per_meaning(BIT_VECTOR_LENGTH, 0);
    GpuVoxelsMapSharedPtr obstacleMapPtr(obstacle);
    size_t collisions = sweptVolume->collideCountingPerMeaning(obstacleMapPtr, collisions_per_meaning);

    //build string to check all swept volume collisions at once
    std::stringstream sstream;
    sstream << "(Meaning|Collisions) ";
    for (size_t i = 0; i < BIT_VECTOR_LENGTH; i++)
    {
      if(collisions_per_meaning.at(i) > 0)
      {
        sstream << "(" << i << "|" << collisions_per_meaning.at(i) << ") ";
      }
    }
    std::string s = sstream.str();

    BOOST_CHECK_MESSAGE(sstream.str() == "(Meaning|Collisions) (14|1) (15|8) (16|8) (17|27) (18|27) (19|27) (20|8) (21|1) (22|1) ", "wrong SweptVolume Parts got hit");
    BOOST_CHECK_MESSAGE(collisions == 108, "collisions == 108");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collide_bitvoxellist_with_countingpermeaning_bitvoxellist", "collide_bitvoxellist_with_countingpermeaning_bitvoxellist", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(collide_bitvoxellist_with_prob_voxelmap)
{
  PERF_MON_START("collide_bitvoxellist_with_prob_voxelmap");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;

    BitVectorVoxelList* list = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    list->insertMetaPointCloud(boxes, voxel_meanings);

    GpuVoxelsMapSharedPtr map_2(new ProbVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP));
    map_2->insertMetaPointCloud(boxes, eBVM_OCCUPIED);

    size_t num_colls = list->collideWith(map_2->as<ProbVoxelMap>(), ProbabilisticVoxel::probabilityToFloat(cSENSOR_MODEL_OCCUPIED));
    BOOST_CHECK_MESSAGE(num_colls == 46, "Number of Collisions == 46");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collide_bitvoxellist_with_prob_voxelmap", "collide_bitvoxellist_with_prob_voxelmap", "voxellists");
  }
}



BOOST_AUTO_TEST_CASE(bitvoxellist_collide_with_types_prob_voxelmap)
{
  PERF_MON_START("bitvoxellist_collide_with_types_prob_voxelmap");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;

    BitVectorVoxelList* list = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    list->insertMetaPointCloud(boxes, voxel_meanings);

    GpuVoxelsMapSharedPtr map_2(new ProbVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP));
    map_2->insertMetaPointCloud(boxes, eBVM_OCCUPIED);

    BitVectorVoxel types_in_collision;

    size_t num_colls = list->collideWithTypes(map_2->as<ProbVoxelMap>(), types_in_collision, 1.0);
    BOOST_CHECK_MESSAGE(num_colls == 46, "Number of Collisions == 46");
    BOOST_CHECK_MESSAGE(types_in_collision.bitVector().getBit(11) && types_in_collision.bitVector().getBit(12), "Both Types found.");
    types_in_collision.bitVector().clearBit(11);
    types_in_collision.bitVector().clearBit(12);
    BOOST_CHECK_MESSAGE(types_in_collision.bitVector().isZero(), "All other Types are clear.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_collide_with_types_prob_voxelmap", "bitvoxellist_collide_with_types_prob_voxelmap", "voxellists");
  }
}


BOOST_AUTO_TEST_CASE(bitvoxellist_collide_with_types_bitvoxelmap)
{
  PERF_MON_START("bitvoxellist_collide_with_types_bitvoxelmap");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;

    BitVectorVoxelList* list = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    list->insertMetaPointCloud(boxes, voxel_meanings);

    GpuVoxelsMapSharedPtr map_2(new BitVectorVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELMAP));
    map_2->insertMetaPointCloud(boxes, eBVM_OCCUPIED);

    BitVectorVoxel types_in_collision;

    size_t num_colls = list->collideWithTypes(map_2->as<BitVectorVoxelMap>(), types_in_collision, 1.0);
    BOOST_CHECK_MESSAGE(num_colls == 46, "Number of Collisions == 46");
    BOOST_CHECK_MESSAGE(types_in_collision.bitVector().getBit(11) && types_in_collision.bitVector().getBit(12), "Both Types found.");
    types_in_collision.bitVector().clearBit(11);
    types_in_collision.bitVector().clearBit(12);
    BOOST_CHECK_MESSAGE(types_in_collision.bitVector().isZero(), "All other Types are clear.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_collide_with_types_bitvoxelmap", "bitvoxellist_collide_with_types_bitvoxelmap", "voxellists");
  }
}


BOOST_AUTO_TEST_CASE(collide_bitvoxellist_with_prob_voxelmap_shifting)
{
  PERF_MON_START("collide_bitvoxellist_with_prob_voxelmap_shifting");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;

    BitVectorVoxelList* list = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);

    std::vector<std::vector<Vector3f> > box_cloud;
    float delta = 0.1;
    box_cloud.push_back(createBoxOfPoints(b1_min, b1_max, delta));

    MetaPointCloud box(box_cloud);
    box.syncToDevice();

    list->insertMetaPointCloud(box, eBVM_OCCUPIED);

    GpuVoxelsMapSharedPtr map_2(new ProbVoxelMap(Vector3ui(dimX, dimY, dimZ), side_length, MT_PROBAB_VOXELMAP));

    size_t num_colls;
    map_2->insertMetaPointCloud(box, eBVM_OCCUPIED);

    for(float shift = 0.0; shift < 4.0; shift += 0.5)
    {
      num_colls = list->collideWith(map_2->as<ProbVoxelMap>(), ProbabilisticVoxel::probabilityToFloat(cSENSOR_MODEL_OCCUPIED), Vector3i(shift, 0,0));
      if(shift < 1.0)
        BOOST_CHECK_MESSAGE(num_colls == 27, "Number of Collisions == 27");
      else if(shift < 2.0)
        BOOST_CHECK_MESSAGE(num_colls == 18, "Number of Collisions == 18");
      else if(shift < 3.0)
        BOOST_CHECK_MESSAGE(num_colls == 9, "Number of Collisions == 9");
      else
        BOOST_CHECK_MESSAGE(num_colls == 0, "Number of Collisions == 0");
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("collide_bitvoxellist_with_prob_voxelmap_shifting", "collide_bitvoxellist_with_prob_voxelmap_shifting", "voxellists");
  }
}


BOOST_AUTO_TEST_CASE(bitvoxellist_insert_metapointcloud)
{
  PERF_MON_START("bitvoxellist_insert_metapointcloud");
  for(int i = 0; i < iterationCount; i++)
  {
    // Generate two boxes that each occupie 27 Voxel ==> 54 Voxels.
    // They overlap each other by 8 Voxels. ==> 46 Voxels should remain in the list.
    // The 8 Voxels that overlap must have set bits of both dense pointcloud meanings!

    BitVectorVoxelList* list = new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    list->insertMetaPointCloud(boxes, voxel_meanings);

    thrust::device_vector<Cube>* d_cubes = NULL;
    list->extractCubes(&d_cubes);
    thrust::host_vector<Cube> h_cubes = *d_cubes;

    BOOST_CHECK_MESSAGE(h_cubes.size() == 46, "Number of reduced cubes == 46");

    BitVector<BIT_VECTOR_LENGTH> ref_bv;
    ref_bv.setBit(11);
    ref_bv.setBit(12);

    size_t num_voxels_with_both_bits = 0;
    for(size_t i = 0; i < h_cubes.size(); i++)
    {
      if(h_cubes[i].m_type_vector == ref_bv)
      {
        num_voxels_with_both_bits++;
        //std::cout << "Cube[" << i << "] bitvector: " << h_cubes[i].m_type_vector << std::endl;
        //std::cout << "Cube[" << i << "] position: " << h_cubes[i].m_position << std::endl;
        bool position_wrong = (h_cubes[i].m_position.x < 1.9 || h_cubes[i].m_position.y < 1.9 || h_cubes[i].m_position.z < 1.9 ||
                               h_cubes[i].m_position.x > 3.1 || h_cubes[i].m_position.y > 3.1 || h_cubes[i].m_position.z > 3.1);
        BOOST_CHECK_MESSAGE(!position_wrong, "Pose of merged voxel-vector in center");
      }
    }
    BOOST_CHECK_MESSAGE(num_voxels_with_both_bits == 8, "Detect 8 overlapping voxels.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_insert_metapointcloud", "bitvoxellist_insert_metapointcloud", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(bitvoxellist_findMatchingVoxels)
{
  // this test depends on the limits of the voxellist, in part because of voxellist out_of_bounds checking
  int dimX = 500;
  int dimY = dimX;
  int dimZ = 200;
  
  PERF_MON_START("bitvoxellist_findMatchingVoxels");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 0.01;

    //create lists
    GpuVoxelsMapSharedPtr voxellist1_shrd_ptr(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST));
    GpuVoxelsMapSharedPtr voxellist2_shrd_ptr(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST));

    //points which are not colliding to bloat the lists
    std::vector<Vector3f> box1 = createBoxOfPoints(Vector3f(2.1, 0.1, 5.1), Vector3f(4.9, 4.9, 9.9), 0.02f);
    std::vector<Vector3f> box2 = createBoxOfPoints(Vector3f(5.1, 5.1, 10.1), Vector3f(9.9, 9.9, 14.9), 0.3f);

    //points actually participating in collision
    std::vector<Vector3f> collisionPoints1;
    for(int j = 0; j < 40; j++)
    {
      collisionPoints1.push_back(Vector3f(j / 40.0 * 5.0, 1.0, 1.0));
    }
    std::vector<Vector3f> collisionPoints2;
    for(int j = 0; j < 30; j++)
    {
      collisionPoints2.push_back(Vector3f(j / 30.0 * 5.0, 2.0, 1.0));
    }
    std::vector<Vector3f> collisionPoints3;
    for(int j = 0; j < 20; j++)
    {
      collisionPoints3.push_back(Vector3f(j / 20.0 * 5.0, 3.0, 1.0));
    }

    //insert ==> voxellist1_shrd_ptr will be the list with more entries
    voxellist1_shrd_ptr->insertPointCloud(box1, BitVoxelMeaning(10));
    voxellist1_shrd_ptr->insertPointCloud(collisionPoints1, BitVoxelMeaning(11));
    voxellist1_shrd_ptr->insertPointCloud(collisionPoints2, BitVoxelMeaning(12));
    voxellist1_shrd_ptr->insertPointCloud(collisionPoints3, BitVoxelMeaning(13));
    voxellist2_shrd_ptr->insertPointCloud(box2, BitVoxelMeaning(20));
    voxellist2_shrd_ptr->insertPointCloud(collisionPoints1, BitVoxelMeaning(21));
    voxellist2_shrd_ptr->insertPointCloud(collisionPoints2, BitVoxelMeaning(22));
    voxellist2_shrd_ptr->insertPointCloud(collisionPoints3, BitVoxelMeaning(23));

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_findMatchingVoxels", "bitvoxellist_findMatchingVoxels::data_generation", "voxellists");

    // Now we collide the lists in different orders and verify the result.
    // The collision checker internally swaps lists due to performance reasons.
    // So we test for if the output contains data from the given input list:

    //first collide list1 with list2
    size_t num_coll = 0;

    std::vector<size_t> collisions_per_meaning(BIT_VECTOR_LENGTH, 0);
    num_coll = voxellist1_shrd_ptr->as<BitVectorVoxelList>()->collideCountingPerMeaning(voxellist2_shrd_ptr, collisions_per_meaning);
    BOOST_CHECK(num_coll == 90);
    BOOST_CHECK(collisions_per_meaning[11] == 40 && collisions_per_meaning[12] == 30 && collisions_per_meaning[13] == 20);
    BOOST_CHECK(collisions_per_meaning[21] == 0 && collisions_per_meaning[22] == 0 && collisions_per_meaning[23] == 0);
    BOOST_CHECK(collisions_per_meaning[10] == 0 && collisions_per_meaning[20] == 0);
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_findMatchingVoxels", "bitvoxellist_findMatchingVoxels::collision", "voxellists");

    //then collide list2 with list1
    collisions_per_meaning.assign(BIT_VECTOR_LENGTH, 0);
    num_coll = voxellist2_shrd_ptr->as<BitVectorVoxelList>()->collideCountingPerMeaning(voxellist1_shrd_ptr, collisions_per_meaning);
    BOOST_CHECK(num_coll == 90);
    BOOST_CHECK(collisions_per_meaning[21] == 40 && collisions_per_meaning[22] == 30 && collisions_per_meaning[23] == 20);
    BOOST_CHECK(collisions_per_meaning[11] == 0 && collisions_per_meaning[12] == 0 && collisions_per_meaning[13] == 0);
    BOOST_CHECK(collisions_per_meaning[10] == 0 && collisions_per_meaning[20] == 0);
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_findMatchingVoxels", "bitvoxellist_findMatchingVoxels::collisionReverse", "voxellists");
  }
}


BOOST_AUTO_TEST_CASE(bitvoxellist_findMatching_omit)
{
  // this test depends on the limits of the voxellist, in part because of voxellist out_of_bounds checking
  int dimX = 500;
  int dimY = dimX;
  int dimZ = 200;
  
  PERF_MON_START("bitvoxellist_findMatching_omit");
  for(int omit = 0; omit <= 1; omit++)
  {
    for(int i = 0; i < iterationCount; i++)
    {
      float side_length = 0.01;

      //create lists
      GpuVoxelsMapSharedPtr voxellist1_shrd_ptr(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST));
      GpuVoxelsMapSharedPtr voxellist2_shrd_ptr(new CountingVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_COUNTING_VOXELLIST));
      GpuVoxelsMapSharedPtr matching_shrd_ptr(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST));

      //points which are not colliding to bloat the lists
      std::vector<Vector3f> box1 = createBoxOfPoints(Vector3f(2.1, 0.1, 5.1), Vector3f(4.9, 4.9, 9.9), 0.02f);
      std::vector<Vector3f> box2 = createBoxOfPoints(Vector3f(5.1, 5.1, 10.1), Vector3f(9.9, 9.9, 14.9), 0.3f);

      //points actually participating in collision
      std::vector<Vector3f> collisionPoints1;
      for(int j = 0; j < 40; j++)
      {
        collisionPoints1.push_back(Vector3f(j / 40.0 * 5.0, 1.0, 1.0));
      }
      std::vector<Vector3f> collisionPoints2;
      for(int j = 0; j < 30; j++)
      {
        collisionPoints2.push_back(Vector3f(j / 30.0 * 5.0, 2.0, 1.0));
      }
      std::vector<Vector3f> collisionPoints3;
      for(int j = 0; j < 20; j++)
      {
        collisionPoints3.push_back(Vector3f(j / 20.0 * 5.0, 3.0, 1.0));
      }

      //insert ==> voxellist1_shrd_ptr will be the list with more entries
      voxellist1_shrd_ptr->insertPointCloud(box1, BitVoxelMeaning(10));
      voxellist1_shrd_ptr->insertPointCloud(collisionPoints1, BitVoxelMeaning(11));
      voxellist1_shrd_ptr->insertPointCloud(collisionPoints2, BitVoxelMeaning(12));
      voxellist1_shrd_ptr->insertPointCloud(collisionPoints3, BitVoxelMeaning(13));
      voxellist2_shrd_ptr->insertPointCloud(box2, BitVoxelMeaning(20));
      voxellist2_shrd_ptr->insertPointCloud(collisionPoints1, BitVoxelMeaning(21));
      voxellist2_shrd_ptr->insertPointCloud(collisionPoints2, BitVoxelMeaning(22));
      voxellist2_shrd_ptr->insertPointCloud(collisionPoints3, BitVoxelMeaning(23));

      PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_findMatching_omit", "bitvoxellist_findMatching_omit::data_generation", "voxellists");

      // Now we collide the lists in different orders and verify the result.
      // The collision checker internally swaps lists due to performance reasons.
      // So we test for if the output contains data from the given input list:
      
      gpu_voxels::Vector3i offset;
      voxellist1_shrd_ptr->as<voxellist::BitVectorVoxelList>()->findMatchingVoxels(
        voxellist2_shrd_ptr->as<voxellist::CountingVoxelList>(),
        offset,
        matching_shrd_ptr->as<voxellist::BitVectorVoxelList>(),
        omit // don't omit coords
      );

      //first collide list1 with list2
      size_t num_coll = matching_shrd_ptr->as<voxellist::BitVectorVoxelList>()->getDimensions().x;
      BOOST_CHECK(num_coll == 90);
      PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_findMatching_omit", "bitvoxellist_findMatching_omit::collision", "voxellists");
    }
  }
}


BOOST_AUTO_TEST_CASE(bitvoxellist_bitshift_collision)
{
  PERF_MON_START("bitvoxellist_bitshift_collision");
  for(int i = 0; i < iterationCount; i++)
  {
    float side_length = 1.f;
    BitVectorVoxelList map_1(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST);
    GpuVoxelsMapSharedPtr map_2(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), side_length, MT_BITVECTOR_VOXELLIST));
    BitVectorVoxelList* map2_base_ptr = dynamic_cast<BitVectorVoxelList*>(map_2.get());

    std::vector<Vector3f> points;
    points.push_back(Vector3f(0.3,0.3,0.3));

    uint32_t shift_size = 0;
    const uint32_t shift_start = 50;

    const uint32_t type_int = eBVM_SWEPT_VOLUME_START + shift_start;
    const BitVoxelMeaning type_2 = BitVoxelMeaning(type_int);

    while (shift_size < shift_start + eBVM_SWEPT_VOLUME_START - 5)
    {
      map_1.clearMap();
      map_2->clearMap();

      map_2->insertPointCloud(points, type_2);
      const BitVoxelMeaning type_1 = BitVoxelMeaning(type_int - shift_size);
      map_1.insertPointCloud(points, type_1);

      map2_base_ptr->shiftLeftSweptVolumeIDs(shift_size);

      size_t num_collisions = 0;
      BitVectorVoxel types_voxel;
      size_t window_size = 1;

      num_collisions = map_1.collideWithBitcheck(map_2->as<voxellist::BitVectorVoxelList>(), window_size);

      if (shift_size <= shift_start)
      {
        BOOST_CHECK(num_collisions == 1);
      }
      else
      {
        BOOST_CHECK(num_collisions == 0);
      }
      shift_size++;
    }
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_bitshift_collision", "bitvoxellist_bitshift_collision", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(voxellist_equals_function)
{
  PERF_MON_START("voxellist_equals_function");
  for(int i = 0; i < iterationCount; i++)
  {
    BitVectorVoxelList list1(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);
    BitVectorVoxelList list2(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    std::vector<Vector3f> outliers;
    outliers.push_back(Vector3f(9.22, 9.22, 9.22));

    list1.insertMetaPointCloud(boxes, voxel_meanings);
    list2.insertMetaPointCloud(boxes, voxel_meanings);
    BOOST_CHECK_MESSAGE(list1.equals(list2), "Lists are equal.");

    list1.insertPointCloud(outliers, BitVoxelMeaning(11));
    list2.insertPointCloud(outliers, BitVoxelMeaning(12));
    BOOST_CHECK_MESSAGE(!list1.equals(list2), "Lists differ.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxellist_equals_function", "voxellist_equals_function", "voxellists");
  }
}


BOOST_AUTO_TEST_CASE(voxellist_disk_io)
{
  PERF_MON_START("voxellist_disk_io");
  for(int i = 0; i < iterationCount; i++)
  {
    BitVectorVoxelList list(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 0.1;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    list.insertMetaPointCloud(boxes, voxel_meanings);

    list.writeToDisk("temp_list.lst");

    BitVectorVoxelList list2(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);

    list2.readFromDisk("temp_list.lst");

    BOOST_CHECK_MESSAGE(list.equals(list2), "List from Disk equals original list.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxellist_disk_io", "voxellist_disk_io", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(bitvoxellist_subtract)
{
  PERF_MON_START("bitvoxellist_subtract");
  for(int i = 0; i < iterationCount; i++)
  {
    // Generate two boxes that each occupie 27 Voxel.
    // They overlap each other by 8 Voxels. Subtract them. ==> 19 Voxels should remain in the list.
    BitVectorVoxelList list1(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);
    BitVectorVoxelList list2(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    float delta = 0.1;
    std::vector<Vector3f> box_cloud1 = createBoxOfPoints(b1_min, b1_max, delta);
    std::vector<Vector3f> box_cloud2 = createBoxOfPoints(b2_min, b2_max, delta);

    list1.insertPointCloud(box_cloud1, BitVoxelMeaning(11));
    list2.insertPointCloud(box_cloud2, BitVoxelMeaning(12));

    list1.subtract(&list2, Vector3f());

    thrust::device_vector<Cube>* d_cubes = NULL;
    list1.extractCubes(&d_cubes);
    thrust::host_vector<Cube> h_cubes = *d_cubes;

    BOOST_CHECK_MESSAGE(h_cubes.size() == 19, "Number of cubes after subtract == 19");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitvoxellist_subtract", "bitvoxellist_subtract", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(countingvoxellist_collide_bitvectorvoxellist_minimal)
{
  PERF_MON_START("countingvoxellist_collide_bitvectorvoxellist_minimal");
  for (int i = 0; i < iterationCount; i++)
  {
    CountingVoxelList list1(Vector3ui(dimX, dimY, dimZ), 1, MT_COUNTING_VOXELLIST);

    Vector3f testPoint1(1.1, 1.1, 1.1);
    Vector3f testPoint2(1.1, 1.1, 1.2);
    Vector3f testPoint3(3.0, 3.0, 3.0);

    std::vector<Vector3f> cloud;
    cloud.push_back(testPoint1);
    cloud.push_back(testPoint2);
    cloud.push_back(testPoint3);

    std::vector<std::vector<Vector3f> > clouds;
    clouds.push_back(cloud);

    MetaPointCloud points(clouds);
    points.syncToDevice();

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));

    list1.insertMetaPointCloud(points, voxel_meanings);

    GpuVoxelsMapSharedPtr map_2(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST));
    map_2->insertMetaPointCloud(points, eBVM_OCCUPIED);

    size_t num_colls1 = list1.collideWith(map_2->as<BitVectorVoxelList>(), 1.0);
    size_t num_colls2 = list1.collideWith(map_2->as<BitVectorVoxelList>(), 2.0);

    BOOST_CHECK_MESSAGE(num_colls1 == 2, "Number of Collisions1 == 2");
    BOOST_CHECK_MESSAGE(num_colls2 == 1, "Number of Collisions2 == 1");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("countingvoxellist_collide_bitvectorvoxellist_minimal", "countingvoxellist_collide_bitvectorvoxellist_minimal", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(countingvoxellist_collide_bitvectorvoxellist)
{
  PERF_MON_START("countingvoxellist_collide_bitvectorvoxellist");
  for (int i = 0; i < iterationCount; i++)
  {
    CountingVoxelList list1(Vector3ui(dimX, dimY, dimZ), 1, MT_COUNTING_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    std::vector<BitVoxelMeaning> voxel_meanings;
    voxel_meanings.push_back(BitVoxelMeaning(11));
    voxel_meanings.push_back(BitVoxelMeaning(12));

    std::vector<std::vector<Vector3f> > box_clouds;
    float delta = 1.0f;

    box_clouds.push_back(createBoxOfPoints(b1_min, b1_max, delta));
    box_clouds.push_back(createBoxOfPoints(b2_min, b2_max, delta));

//    for (int i=0; i<box_clouds.size(); i++) {
//        std::cout << "start vector " << std::endl;
//        for (int j=0; j<box_clouds[i].size(); j++) {
//            std::cout << box_clouds[i][j] << std::endl;
//        }
//    }

    MetaPointCloud boxes(box_clouds);
    boxes.syncToDevice();

    list1.insertMetaPointCloud(boxes, voxel_meanings);

    GpuVoxelsMapSharedPtr map_2(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST));
    map_2->insertMetaPointCloud(boxes, eBVM_OCCUPIED);

//    //DEBUG
//    list1.screendump(true);
//    map_2->as<BitVectorVoxelList>()->screendump(true);

    size_t num_colls = list1.collideWith(map_2->as<BitVectorVoxelList>(), 1.0);

//    //DEBUG
//    list1.screendump(true);
//    map_2->as<BitVectorVoxelList>()->screendump(true);

    BOOST_CHECK_MESSAGE(num_colls == 46, "Number of Collisions == 46");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("countingvoxellist_collide_bitvectorvoxellist", "countingvoxellist_collide_bitvectorvoxellist", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(countingvoxellist_subtract_bitvectorvoxellist_minimal)
{
  PERF_MON_START("countingvoxellist_subtract_bitvectorvoxellist_minimal");
  for (int i = 0; i < iterationCount; i++)
  {
    GpuVoxelsMapSharedPtr list1(new CountingVoxelList(Vector3ui(dimX, dimY, dimZ), 1, MT_COUNTING_VOXELLIST));
    GpuVoxelsMapSharedPtr list2(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST));

    Vector3f testPoint1(1.1, 1.1, 1.1);
    Vector3f testPoint2(1.2, 1.1, 1.1);
    Vector3f testPoint3(3.1, 3.1, 3.1);
    Vector3f testPoint4(5.0, 5.0, 5.0);

    std::vector<Vector3f> cloud1;
    cloud1.push_back(testPoint1);
    cloud1.push_back(testPoint2);
    cloud1.push_back(testPoint3);
    cloud1.push_back(testPoint4);

    std::vector<Vector3f> cloud2;
    cloud2.push_back(testPoint1);
    cloud2.push_back(testPoint3);

    list1->insertPointCloud(cloud1, BitVoxelMeaning(11));
    list2->insertPointCloud(cloud2, BitVoxelMeaning(12));

    list1->as<CountingVoxelList>()->subtractFromCountingVoxelList(list2->as<BitVectorVoxelList>(), Vector3f());

    thrust::device_vector<Cube> *d_cubes = NULL;
    list1->as<CountingVoxelList>()->extractCubes(&d_cubes);
    thrust::host_vector<Cube> h_cubes = *d_cubes;

    BOOST_CHECK_MESSAGE(h_cubes.size() == 1, "Number of cubes after subtract == 1 ");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("countingvoxellist_subtract_bitvectorvoxellist_minimal", "countingvoxellist_subtract_bitvectorvoxellist_minimal", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(countingvoxellist_merge_into_bitvectorvoxellist_minimal)
{
  PERF_MON_START("countingvoxellist_merge_into_bitvectorvoxellist_minimal");
  for (int i = 0; i < iterationCount; i++)
  {
    GpuVoxelsMapSharedPtr list1(new CountingVoxelList(Vector3ui(dimX, dimY, dimZ), 0.1f, MT_COUNTING_VOXELLIST));
    GpuVoxelsMapSharedPtr list2(new BitVectorVoxelList(Vector3ui(dimX, dimY, dimZ), 0.1f, MT_BITVECTOR_VOXELLIST));

    Vector3f testPoint1(1.1, 1.1, 1.1);
    Vector3f testPoint2(1.2, 1.1, 1.1);
    Vector3f testPoint3(3.1, 3.1, 3.1);
    Vector3f testPoint4(5.0, 5.0, 5.0);

    std::vector<Vector3f> cloud1;
    cloud1.push_back(testPoint1);
    cloud1.push_back(testPoint2);
    cloud1.push_back(testPoint4);

    std::vector<Vector3f> cloud2;
    cloud2.push_back(testPoint1);
    cloud2.push_back(testPoint3);

    list1->insertPointCloud(cloud1, BitVoxelMeaning(11));
    list2->insertPointCloud(cloud2, BitVoxelMeaning(12));

    BitVoxelMeaning bvm = eBVM_OCCUPIED;
    list2->merge(list1, Vector3f(), &bvm);

    // list2->as<BitVectorVoxelList>()->screendump(true); //DEBUG

    thrust::device_vector<Cube> *d_cubes = NULL;
    list2->as<BitVectorVoxelList>()->extractCubes(&d_cubes);
    thrust::host_vector<Cube> h_cubes = *d_cubes;

    BOOST_CHECK_MESSAGE(h_cubes.size() == 4, "Number of cubes after merge == 4 ");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("countingvoxellist_merge_into_bitvectorvoxellist_minimal", "countingvoxellist_merge_into_bitvectorvoxellist_minimal", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(countingvoxellist_subtract_bitvectorvoxellist)
{
  PERF_MON_START("countingvoxellist_subtract_bitvectorvoxellist");
  for (int i = 0; i < iterationCount; i++)
  {
    CountingVoxelList list1(Vector3ui(dimX, dimY, dimZ), 1, MT_COUNTING_VOXELLIST);
    // BitVectorVoxelList list1(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);
    BitVectorVoxelList list2(Vector3ui(dimX, dimY, dimZ), 1, MT_BITVECTOR_VOXELLIST);

    Vector3f b1_min(1.1,1.1,1.1);
    Vector3f b1_max(3.9,3.9,3.9);
    Vector3f b2_min(2.1,2.1,2.1);
    Vector3f b2_max(4.9,4.9,4.9);

    float delta = 0.1;
    std::vector<Vector3f> box_cloud1 = createBoxOfPoints(b1_min, b1_max, delta);
    std::vector<Vector3f> box_cloud2 = createBoxOfPoints(b2_min, b2_max, delta);

    list1.insertPointCloud(box_cloud1, BitVoxelMeaning(11));
    list2.insertPointCloud(box_cloud2, BitVoxelMeaning(12));

    //list1.subtract(&list2, Vector3f());
    list1.subtractFromCountingVoxelList(&list2, Vector3f());

    thrust::device_vector<Cube>* d_cubes = NULL;
    list1.extractCubes(&d_cubes);
    thrust::host_vector<Cube> h_cubes = *d_cubes;

    BOOST_CHECK_MESSAGE(h_cubes.size() == 19, "Number of cubes after subtract == 19");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("countingvoxellist_subtract_bitvectorvoxellist", "countingvoxellist_subtract_bitvectorvoxellist", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(bitmasked_collision)
{
  PERF_MON_START("bitmasked_collision");
  for(int i = 0; i < iterationCount; i++)
  {
    /*
   *    Generate three boxes that each occupie 64 Voxel. They lie along the X-axis next to each other and overlap with their neighbour by 16 Voxels.
   *    Generate another bigger box that intersects with the three smaller boxes about 2 rows on the Y-axis (72 Voxels intersecting).
   *
   *
   *       Voxellist:                     Voxelmap:
   *       / = Type 1                     o = Occupied
   *       \ = Type 2
   *       X = Type 1&2
   *    y                              y
   *    ^                              ^
   *    |                              |
   *    |  ---------------             |
   *    |  |/|/|X|\|X|/|/|             |
   *    |  ---------------             |
   *    |  |/|/|X|\|X|/|/|             |
   *    |  ---------------             |  ---------------
   *    |  |/|/|X|\|X|/|/|             |  |o|o|o|o|o|o|o|
   *    |  ---------------             |  ---------------
   *    |                              |  |o|o|o|o|o|o|o|
   *    |                              |  ---------------
   *    |                              |  |o|o|o|o|o|o|o|
   *    |                              |  ---------------
   *    ----------------------> x      ----------------------> x
   *
   */

    // ==> To get some more CUDA Blocks involved,
    // we have to scale down the voxelsize to 1/4 ==> Voxels in collision * 64
    BitVectorVoxelList list(Vector3ui(dimX, dimY, dimZ), 0.025, MT_BITVECTOR_VOXELLIST);
    ProbVoxelMap map(Vector3ui(dimX, dimY, dimZ), 0.025, MT_PROBAB_VOXELMAP);

    float delta = 0.005;
    Vector3f b_min(0.111,0.111,0.111);
    Vector3f b_max(0.799,0.399,0.399);
    std::vector<Vector3f> box_cloud = createBoxOfPoints(b_min, b_max, delta);
    map.insertPointCloud(box_cloud, gpu_voxels::eBVM_OCCUPIED);

    b_min = Vector3f(0.111,0.311,0.111);
    b_max = Vector3f(0.399,0.599,0.399);
    box_cloud = createBoxOfPoints(b_min, b_max, delta);
    list.insertPointCloud(box_cloud, gpu_voxels::BitVoxelMeaning(34));

    b_min = Vector3f(0.311,0.311,0.111);
    b_max = Vector3f(0.599,0.599,0.399);
    box_cloud = createBoxOfPoints(b_min, b_max, delta);
    list.insertPointCloud(box_cloud, gpu_voxels::BitVoxelMeaning(63));

    b_min = Vector3f(0.511,0.311,0.111);
    b_max = Vector3f(0.799,0.599,0.399);
    box_cloud = createBoxOfPoints(b_min, b_max, delta);
    list.insertPointCloud(box_cloud, gpu_voxels::BitVoxelMeaning(102));

    size_t num_colls;
    BitVectorVoxel types_to_check;

    // No offset given:
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f);
    BOOST_CHECK_MESSAGE(num_colls == 0, "Zero collisions expected");

    types_to_check.bitVector().setBit(33);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f);
    BOOST_CHECK_MESSAGE(num_colls == 0, "Zero collisions expected");

    types_to_check.bitVector().setBit(34);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f);
    BOOST_CHECK_MESSAGE(num_colls == 9*64, "9*64 collisions expected");

    types_to_check.bitVector().setBit(63);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f);
    BOOST_CHECK_MESSAGE(num_colls == 15*64, "15*64 collisions expected");

    types_to_check.bitVector().setBit(102);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f);
    BOOST_CHECK_MESSAGE(num_colls == 21*64, "21*64 collisions expected");

    // Offset shifting List on Y-Axis into Map:
    Vector3i offset(0,-4,0);
    types_to_check.bitVector().clear();
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 0, "Shift (0 -4 0): Zero collisions expected");

    types_to_check.bitVector().setBit(33);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 0, "Shift (0 -4 0): Zero collisions expected");

    types_to_check.bitVector().setBit(34);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 18*64, "Shift (0 -4 0): 18*64 collisions expected");

    types_to_check.bitVector().setBit(63);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 30*64, "Shift (0 -4 0): 30*64 collisions expected");

    types_to_check.bitVector().setBit(102);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 42*64, "Shift (0 -4 0): 42*64 collisions expected");

    // Offset shifting List on Y-Axis into Map and on Z out of map
    offset = Vector3i(0,-4,-4);
    types_to_check.bitVector().clear();
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 0, "Shift (0 -4 -4): Zero collisions expected");

    types_to_check.bitVector().setBit(33);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 0, "Shift (0 -4 -4): Zero collisions expected");

    types_to_check.bitVector().setBit(34);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 12*64, "Shift (0 -4 -4): 12*64 collisions expected");

    types_to_check.bitVector().setBit(63);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 20*64, "Shift (0 -4 -4): 24*64 collisions expected");

    types_to_check.bitVector().setBit(102);
    num_colls = list.collideWithTypeMask(&map, types_to_check, 1.0f, offset);
    BOOST_CHECK_MESSAGE(num_colls == 28*64, "Shift (0 -4 -4): 36*64 collisions expected");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("bitmasked_collision", "bitmasked_collision", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(countingvoxellist_remove_underpopulated)
{
  PERF_MON_START("countingvoxellist_remove_underpopulated");

  std::vector<Vector3f> listPoints;
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.5f));

  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.4f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.4f));

  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.3f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.3f));
  listPoints.push_back(Vector3f(0.2f, 0.7f, 0.3f));

  for (int i = 0; i < iterationCount; i++)
  {
    CountingVoxelList list1(Vector3ui(dimX, dimY, dimZ), 0.01, MT_COUNTING_VOXELLIST);

    // insert an obstacle to set each Counting voxel to 1
    list1.insertPointCloud(listPoints, gpu_voxels::eBVM_OCCUPIED);

    // TODO: fix insertion error that leads to unexpected voxel content of 6, 4 and 2!

    size_t remaining = list1.m_dev_id_list.size();
    BOOST_CHECK_MESSAGE(remaining == 3, "There are 3 voxels at first");

    // remove underpopulated voxels: should result in one empty voxellist and two untouched ones
    list1.remove_underpopulated(0);

    remaining = list1.m_dev_id_list.size();
    BOOST_CHECK_MESSAGE(remaining == 3, "All 3 voxels have a count of more than 0");

    // remove underpopulated voxels: should result in one empty voxellist and two untouched ones
    list1.remove_underpopulated(2);

    remaining = list1.m_dev_id_list.size();
    BOOST_CHECK_MESSAGE(remaining == 2, "Only 2 voxels have a count of at least 2");

    list1.screendump(true);

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("countingvoxellist_remove_underpopulated", "countingvoxellist_remove_underpopulated", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(voxellist_copy_coords_to_host)
{
  PERF_MON_START("voxellist_copy_coords_to_host");

  float side_length = 0.1f;
  float delta = side_length;

  Vector3f box1_min = Vector3f(0.9,0.9,0.9);
  Vector3f box1_max = Vector3f(1.1,1.1,1.1);
  std::vector<Vector3f> box1 = geometry_generation::createBoxOfPoints(box1_min, box1_max, delta);

  Vector3f box2_min = Vector3f(0.3,0.3,0.3);
  Vector3f box2_max = Vector3f(0.5,0.5,0.5);
  std::vector<Vector3f> box2 = geometry_generation::createBoxOfPoints(box2_min, box2_max, delta);

  for (int i = 0; i < iterationCount; i++)
  {
    BitVectorVoxelList list(Vector3ui(16, 16, 16), side_length, MT_BITVECTOR_VOXELLIST);

    list.insertPointCloud(box1, (BitVoxelMeaning) 20);
    list.insertPointCloud(box2, (BitVoxelMeaning) 40);

    // Copy coords with bits 15..25 to host (i.e. box1)
    std::vector<Vector3ui> filtered_coordinates;
    list.copyCoordsToHostBvmBounded(filtered_coordinates, (BitVoxelMeaning) 15, (BitVoxelMeaning) 25);

    // Check that filtered coordinates are within bounds of box1
    for (size_t i = 0; i < filtered_coordinates.size(); i++)
    {
      Vector3f point;
      point.x = filtered_coordinates[i].x * side_length;
      point.y = filtered_coordinates[i].y * side_length;
      point.z = filtered_coordinates[i].z * side_length;
      BOOST_CHECK_MESSAGE(box1_min.x <= point.x && point.x <= box1_max.x && box1_min.y <= point.y && point.y <= box1_max.y && box1_min.z <= point.z && point.z <= box1_max.z, "Invalid point " << point.x << ", " << point.y << ", " << point.z);
    }

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxellist_copy_coords_to_host", "voxellist_copy_coords_to_host", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(voxellist_merge_bvm)
{
  PERF_MON_START("voxellist_merge_bvm");

  float side_length = 0.1f;
  float delta = side_length;

  for (int i = 0; i < iterationCount; i++)
  {
    BitVectorVoxelList list(Vector3ui(16, 16, 16), side_length, MT_BITVECTOR_VOXELLIST);
    boost::shared_ptr<BitVectorVoxelMap> map = boost::make_shared<BitVectorVoxelMap>(Vector3ui(16, 16, 16), side_length, MT_BITVECTOR_VOXELMAP);

    // Insert box coordinates into map
    Vector3f box_min = Vector3f(0.9,0.9,0.9);
    Vector3f box_max = Vector3f(1.1,1.1,1.1);
    std::vector<Vector3ui> box_coordinates = geometry_generation::createBoxOfPoints(box_min, box_max, delta, side_length);
    map->insertCoordinateList(box_coordinates, eBVM_OCCUPIED);

    // Merge into list
    list.merge(map, Vector3i());

    // Copy from list to host
    std::vector<Vector3ui> filtered_coordinates;
    list.copyCoordsToHost(filtered_coordinates);

    // We should now have two equal sets of coordinates (possibly in different orders)
    BOOST_REQUIRE(filtered_coordinates.size() == box_coordinates.size());
    while (!filtered_coordinates.empty())
    {
      Vector3ui el = filtered_coordinates.back();
      filtered_coordinates.pop_back();

      std::vector<Vector3ui>::iterator correspondence = std::find(box_coordinates.begin(), box_coordinates.end(), el);
      BOOST_REQUIRE(correspondence != box_coordinates.end());
      box_coordinates.erase(correspondence);
    }

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxellist_merge_bvm", "voxellist_merge_bvm", "voxellists");
  }
}

BOOST_AUTO_TEST_CASE(voxellist_cloning)
{
  PERF_MON_START("voxellist_cloning");

  float side_length = 0.1f;
  float delta = side_length;

  for (int i = 0; i < iterationCount; i++)
  {
    boost::shared_ptr<BitVectorVoxelList> list1 = boost::make_shared<BitVectorVoxelList>(Vector3ui(16, 16, 16), side_length, MT_BITVECTOR_VOXELLIST);
    boost::shared_ptr<BitVectorVoxelList> list2 = boost::make_shared<BitVectorVoxelList>(Vector3ui(16, 16, 16), side_length, MT_BITVECTOR_VOXELLIST);
    boost::shared_ptr<BitVectorVoxelList> list3 = boost::make_shared<BitVectorVoxelList>(Vector3ui(32, 16, 16), side_length, MT_BITVECTOR_VOXELLIST);

    // Insert box coordinates into list1
    Vector3f box_min = Vector3f(0.9,0.9,0.9);
    Vector3f box_max = Vector3f(1.1,1.1,1.1);
    std::vector<Vector3ui> box_coordinates = geometry_generation::createBoxOfPoints(box_min, box_max, delta, side_length);
    list1->insertCoordinateList(box_coordinates, eBVM_OCCUPIED);

    // Merge into list2 and list3
    list2->clone(*list1);
    list3->clone(*list1);

    // Check list2 for equality
    std::vector<Vector3ui> cloned_coordinates;
    list2->copyCoordsToHost(cloned_coordinates);

    BOOST_REQUIRE_MESSAGE(cloned_coordinates.size() == box_coordinates.size(), "Cloned invalid number of coordinates");
    while (!cloned_coordinates.empty())
    {
      Vector3ui el = cloned_coordinates.back();
      cloned_coordinates.pop_back();

      std::vector<Vector3ui>::iterator correspondence = std::find(box_coordinates.begin(), box_coordinates.end(), el);
      BOOST_REQUIRE_MESSAGE(correspondence != box_coordinates.end(), "Coordinate [" <<  el.x << ", " << el.y << ", " << el.z << "] missing in cloned coordinates");
      box_coordinates.erase(correspondence);
    }

    // Check that list3 is empty (nothing should be cloned here since dimensions are not equal)
    list3->copyCoordsToHost(cloned_coordinates);
    BOOST_CHECK_MESSAGE(cloned_coordinates.empty(), "Copied coordinates when dimensions were invalid");

    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("voxellist_cloning", "voxellist_cloning", "voxellists");
  }
}

BOOST_AUTO_TEST_SUITE_END()


