// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Andreas Hermann <hermann@fzi.de>
 * \date    2015-10-14
 *
 */
//----------------------------------------------------------------------

#include <boost/test/unit_test.hpp>
#include <thrust/host_vector.h>

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/test/Tests.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/octree/GvlNTree.h>
#include <gpu_voxels/voxellist/BitVoxelList.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>

using namespace gpu_voxels;
using namespace NTree;
using namespace geometry_generation;

BOOST_AUTO_TEST_SUITE(octree_collisions)

BOOST_AUTO_TEST_CASE(octree_colliding_new_morton_voxellist)
{

  GvlNTreeDet* my_octree = new GvlNTreeDet(1, MT_BITVECTOR_OCTREE);

  gpu_voxels::voxellist::BitVectorMortonVoxelList* my_voxellist = new gpu_voxels::voxellist::BitVectorMortonVoxelList(Vector3ui(100, 100, 100), 1.0, MT_BITVECTOR_VOXELLIST);

  // Create two overlapping boxes with 6^3 = 216 overlapping voxels.
  std::vector<Vector3f> boxpoints1 = createBoxOfPoints(Vector3f(20, 20, 20), Vector3f(30, 30, 30), 1.0);
  std::vector<Vector3f> boxpoints2 = createBoxOfPoints(Vector3f(25, 25, 25), Vector3f(35, 35, 35), 1.0);

  my_octree->insertPointCloud(boxpoints1, eBVM_OCCUPIED);
  my_voxellist->insertPointCloud(boxpoints2, eBVM_OCCUPIED);


  size_t num_colls;

  num_colls = my_octree->intersect_morton<true, false, false, BitVectorVoxel>(*my_voxellist);

  std::cout << "Num colls: " << num_colls << std::endl;

  BOOST_CHECK_MESSAGE(num_colls == 216, "All collisions detected.");

}

BOOST_AUTO_TEST_CASE(octree_colliding_regular_voxellist)
{

  GvlNTreeDet* my_octree = new GvlNTreeDet(1, MT_BITVECTOR_OCTREE);

  gpu_voxels::voxellist::BitVectorVoxelList* my_voxellist = new gpu_voxels::voxellist::BitVectorVoxelList(Vector3ui(100, 100, 100), 1.0, MT_BITVECTOR_VOXELLIST);


  // Create two overlapping boxes with 6^3 = 216 overlapping voxels.
  std::vector<Vector3f> boxpoints1 = createBoxOfPoints(Vector3f(20, 20, 20), Vector3f(30, 30, 30), 1.0);
  std::vector<Vector3f> boxpoints2 = createBoxOfPoints(Vector3f(25, 25, 25), Vector3f(35, 35, 35), 1.0);

  my_octree->insertPointCloud(boxpoints1, eBVM_OCCUPIED);

  my_voxellist->insertPointCloud(boxpoints2, eBVM_OCCUPIED);


  size_t num_colls;

  num_colls = my_octree->intersect_sparse<true, false, false, BitVectorVoxel>(*my_voxellist);

  std::cout << "Num colls: " << num_colls << std::endl;

  BOOST_CHECK_MESSAGE(num_colls == 216, "All collisions detected.");

}

BOOST_AUTO_TEST_SUITE_END()


