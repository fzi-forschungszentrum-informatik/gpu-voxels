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
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>

#include <boost/test/unit_test.hpp>
#include "helpers.h"

using namespace gpu_voxels;
using namespace voxellist;

BOOST_AUTO_TEST_SUITE(voxellists)

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

    num_collisions = map_1.collideWithBitcheck(map_2, window_size);

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

BOOST_AUTO_TEST_SUITE_END()


