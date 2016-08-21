// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann <hermann@fzi.de>
 * \date    2015-09-22
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/voxellist/BitVoxelList.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>

#include <boost/test/unit_test.hpp>

using namespace gpu_voxels;
using namespace voxellist;
using namespace voxelmap;
using namespace geometry_generation;

BOOST_AUTO_TEST_SUITE(voxellists)

BOOST_AUTO_TEST_CASE(collide_bitvoxellist_with_prob_voxelmap)
{
  Vector3ui dim(103, 123, 105);
  float side_length = 1.f;

  BitVectorVoxelList* list = new BitVectorVoxelList(dim, side_length, MT_BITVECTOR_VOXELLIST);

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

  GpuVoxelsMapSharedPtr map_2(new ProbVoxelMap(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP));
  map_2->insertMetaPointCloud(boxes, eBVM_OCCUPIED);

  size_t num_colls = list->collideWith(map_2->as<ProbVoxelMap>(), 1.0);
  BOOST_CHECK_MESSAGE(num_colls == 46, "Number of Collisions == 46");

}

BOOST_AUTO_TEST_CASE(collide_bitvoxellist_with_prob_voxelmap_shifting)
{
  Vector3ui dim(103, 123, 105);
  float side_length = 1.f;

  BitVectorVoxelList* list = new BitVectorVoxelList(dim, side_length, MT_BITVECTOR_VOXELLIST);

  Vector3f b1_min(1.1,1.1,1.1);
  Vector3f b1_max(3.9,3.9,3.9);

  std::vector<std::vector<Vector3f> > box_cloud;
  float delta = 0.1;
  box_cloud.push_back(createBoxOfPoints(b1_min, b1_max, delta));

  MetaPointCloud box(box_cloud);
  box.syncToDevice();

  list->insertMetaPointCloud(box, eBVM_OCCUPIED);

  GpuVoxelsMapSharedPtr map_2(new ProbVoxelMap(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP));

  size_t num_colls;
  map_2->insertMetaPointCloud(box, eBVM_OCCUPIED);

  for(float shift = 0.0; shift < 4.0; shift += 0.5)
  {
    num_colls = list->collideWith(map_2->as<ProbVoxelMap>(), 1.0, Vector3ui(shift, 0,0));
    if(shift < 1.0)
      BOOST_CHECK_MESSAGE(num_colls == 27, "Number of Collisions == 27");
    else if(shift < 2.0)
      BOOST_CHECK_MESSAGE(num_colls == 18, "Number of Collisions == 18");
    else if(shift < 3.0)
      BOOST_CHECK_MESSAGE(num_colls == 9, "Number of Collisions == 9");
    else
      BOOST_CHECK_MESSAGE(num_colls == 0, "Number of Collisions == 0");
  }
}


BOOST_AUTO_TEST_CASE(bitvoxellist_insert_metapointcloud)
{
  // Generate two boxes that each occupie 27 Voxel ==> 54 Voxels.
  // They overlap each other by 8 Voxels. ==> 46 Voxels should remain in the list.
  // The 8 Voxels that overlap must have set bits of both dense pointcloud meanings!

  BitVectorVoxelList* list = new BitVectorVoxelList(
        Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);

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
}

BOOST_AUTO_TEST_CASE(bitvoxellist_bitshift_collision)
{
  float side_length = 1.f;
  BitVectorVoxelList map_1(Vector3ui(100, 100, 100), side_length, MT_BITVECTOR_VOXELLIST);
  GpuVoxelsMapSharedPtr map_2(new BitVectorVoxelList(Vector3ui(100, 100, 100), side_length, MT_BITVECTOR_VOXELLIST));
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
}

BOOST_AUTO_TEST_CASE(voxellist_equals_function)
{
  BitVectorVoxelList list1( Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);
  BitVectorVoxelList list2(Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);

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
}


BOOST_AUTO_TEST_CASE(voxellist_disk_io)
{
  BitVectorVoxelList list( Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);

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

  BitVectorVoxelList list2(Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);

  list2.readFromDisk("temp_list.lst");

  BOOST_CHECK_MESSAGE(list.equals(list2), "List from Disk equals original list.");
}


BOOST_AUTO_TEST_CASE(bitvoxellist_subtract)
{
  // Generate two boxes that each occupie 27 Voxel.
  // They overlap each other by 8 Voxels. Subtract them. ==> 19 Voxels should remain in the list.
  BitVectorVoxelList list1(Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);
  BitVectorVoxelList list2(Vector3ui(100, 100, 100), 1, MT_BITVECTOR_VOXELLIST);

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

}

BOOST_AUTO_TEST_SUITE_END()


