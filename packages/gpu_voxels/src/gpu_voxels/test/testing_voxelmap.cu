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

#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/voxelmap/Tests.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/voxel/SVCollider.hpp>
#include <gpu_voxels/voxel/BitVoxel.hpp>
#include <gpu_voxels/helpers/GeometryGeneration.h>

#include <boost/test/unit_test.hpp>

using namespace gpu_voxels;
using namespace voxelmap;
using namespace geometry_generation;

BOOST_AUTO_TEST_SUITE(voxelmap)


BOOST_AUTO_TEST_CASE(addresScheme)
{
  size_t nr_of_tests = 30000;
  Vector3ui dim(150, 200, 123);
  float voxel_side_length = 1.f;
  bool success = true;

  test::triggerAddressingTest<BitVectorVoxel>(dim, voxel_side_length, nr_of_tests, &success);

  BOOST_CHECK_MESSAGE(success, "FloatCoords to Voxel == Voxel to FloatCoords");
}


BOOST_AUTO_TEST_CASE(offset)
{
  // (offset.z * (int32_t)dimensions.x * (int32_t)dimensions.y + offset.y * (int32_t)dimensions.x + offset.x);

  Vector3ui dim(15, 16, 17);
  Vector3i  offset(0, -2, 0);
  ptrdiff_t ptr_offset = getVoxelIndexSigned(dim, offset);
  ptrdiff_t ground_truth = -30;
  BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Negative Offset -30 correct.");

  offset = Vector3i(1, -2, -3);
  ptr_offset = getVoxelIndexSigned(dim, offset);
  ground_truth = -749;
  BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Negative Offset -749 correct.");

  offset = Vector3i(3, -2, -1);
  ptr_offset = getVoxelIndexSigned(dim, offset);
  ground_truth = -267;
  BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Negative Offset -267 correct.");

  offset = Vector3i(-1, -2, 3);
  ptr_offset = getVoxelIndexSigned(dim, offset);
  ground_truth = 689;
  BOOST_CHECK_MESSAGE( ptr_offset == ground_truth, "Positive Offset 689 correct.");
}


BOOST_AUTO_TEST_CASE(collision)
{
  size_t nr_tests = 10000;
  Vector3ui dim(89, 123, 74);
  float side_length = 1.f;
  ProbVoxelMap map_1(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP);
  ProbVoxelMap map_2(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP);

  std::vector<Vector3f> this_testpoints;

  createEquidistantPointsInBox(nr_tests, dim, side_length, this_testpoints);

  map_1.insertPointCloud(this_testpoints, eBVM_OCCUPIED);
  map_2.insertPointCloud(this_testpoints, eBVM_OCCUPIED);

  BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1) == nr_tests, "All collisions detected.");
}

//! Create two 3x3x3 boxes that overlap by 8 voxels.
//! Then collide with and without offset.
BOOST_AUTO_TEST_CASE(collision_with_offset)
{
  Vector3ui dim(89, 123, 74);
  float side_length = 1.f;
  ProbVoxelMap map_1(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP);
  ProbVoxelMap map_2(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP);

  std::vector<Vector3f> this_testpoints1;
  std::vector<Vector3f> this_testpoints2;

  this_testpoints1 = createBoxOfPoints( Vector3f(2.1, 2.1, 2.1), Vector3f(4.1, 4.1, 4.1), 0.5);
  this_testpoints2 = createBoxOfPoints( Vector3f(3.1, 3.1, 3.1), Vector3f(5.1, 5.1, 5.1), 0.5);

  map_1.insertPointCloud(this_testpoints1, eBVM_OCCUPIED);
  map_2.insertPointCloud(this_testpoints2, eBVM_OCCUPIED);

  BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1) == 8, "All collisions without offset detected.");
  BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1, Vector3i(-1,0,-1)) == 18, "All collisions with negative offset detected.");
  BOOST_CHECK_MESSAGE(map_2.collideWith(&map_1, 0.1, Vector3i(1,0,1)) == 18, "All collisions with positive offset detected.");

}

BOOST_AUTO_TEST_CASE(no_collision)
{
  size_t nr_tests = 10000;
  Vector3ui dim(89, 123, 74);
  float side_length = 1.f;

  std::vector<Vector3f> this_testpoints;
  std::vector<Vector3f> other_testpoints;

  ProbVoxelMap map_1(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP);
  ProbVoxelMap map_2(dim.x, dim.y, dim.z, side_length, MT_PROBAB_VOXELMAP);

  // clear the map, insert non overlapping point pattern in both maps,
  // check for NO collision
  createNonOverlapping3dCheckerboard(nr_tests, dim, side_length, this_testpoints, other_testpoints);

  map_1.insertPointCloud(this_testpoints, eBVM_OCCUPIED);
  map_2.insertPointCloud(other_testpoints, eBVM_OCCUPIED);

  // there shouldn't be a collision in this test!
  BOOST_CHECK_MESSAGE(map_1.collideWith(&map_2, 0.1) == 0, "No collisions detected.");
}

/**
 * @brief Test for windowed collsion checks.
 *
 */
BOOST_AUTO_TEST_CASE(collision_bitshift)
{
  Vector3ui dim(10, 10, 10);
  float side_length = 1.f;
  BitVectorVoxelMap map_1(dim.x, dim.y, dim.z, side_length, MT_BITVECTOR_VOXELMAP);
  BitVectorVoxelMap map_2(dim.x, dim.y, dim.z, side_length, MT_BITVECTOR_VOXELMAP);

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
  uint32_t dim_x = 10, dim_y = 10, dim_z = 10;
  float side_length = 1.f;
  BitVectorVoxelMap map_1(dim_x, dim_y, dim_z, side_length, MT_BITVECTOR_VOXELMAP);
  BitVectorVoxelMap map_2(dim_x, dim_y, dim_z, side_length, MT_BITVECTOR_VOXELMAP);

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
}

BOOST_AUTO_TEST_CASE(iostream_bitvoxel)
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
}

BOOST_AUTO_TEST_SUITE_END()



