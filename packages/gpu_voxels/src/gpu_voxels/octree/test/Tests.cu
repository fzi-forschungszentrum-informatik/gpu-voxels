// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-01-21
 *
 */
//----------------------------------------------------------------------

#include "Tests.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/set_operations.h>
#include <cuda_runtime.h>

#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/octree/PointCloud.h>
#include <gpu_voxels/helpers/cuda_handling.h>
#include <cuda_profiler_api.h>
#include <vector>
#include <math.h>
#include "Helper.h"
#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/EnvNodesProbabilistic.h>

namespace gpu_voxels {
namespace NTree {
namespace Test {

#ifdef PROBABILISTIC_TREE
  typedef Environment::InnerNodeProb InnerNode;
  typedef Environment::LeafNodeProb LeafNode;
#else
  typedef Environment::InnerNode InnerNode;
  typedef Environment::LeafNode LeafNode;
#endif

//struct Random_generator
//{
//  voxel_id sideLengthInVoxel;
//  unsigned seed;
//
//  __host__ __device__ unsigned int hash(unsigned int a)
//  {
//    a = (a + 0x7ed55d16) + (a << 12);
//    a = (a ^ 0xc761c23c) ^ (a >> 19);
//    a = (a + 0x165667b1) + (a << 5);
//    a = (a + 0xd3a2646c) ^ (a << 9);
//    a = (a + 0xfd7046c5) + (a << 3);
//    a = (a ^ 0xb55a4f09) ^ (a >> 16);
//    return a;
//  }
//
//  __host__ __device__ __forceinline__ voxel_id operator()()
//  {
//    unsigned seed = hash(blockIdx.x * blockDim.x + threadIdx.x);
//    thrust::default_random_engine rng(seed + this->seed);
//    thrust::random::uniform_real_distribution<float> distrib;
//    return (voxel_id) (distrib(rng) * (sideLengthInVoxel * sideLengthInVoxel * sideLengthInVoxel));
//  }
//};
//
//Random_generator rnd;

thrust::host_vector<gpu_voxels::Vector3ui> randomPoints(voxel_count num_points, OctreeVoxelID maxValue)
{
  uint32_t max_coordinate = (uint32_t) ceil(pow(maxValue, 1.0 / 3));
  thrust::host_vector<gpu_voxels::Vector3ui> points(num_points);
  for (voxel_count i = 0; i < num_points; ++i)
  {
    points[i].x = (uint32_t) (drand48() * (max_coordinate - 1));
    points[i].y = (uint32_t) (drand48() * (max_coordinate - 1));
    points[i].z = (uint32_t) (drand48() * (max_coordinate - 1));
  }
  return points;
}

thrust::host_vector<Voxel> randomVoxel(OctreeVoxelID num_points, OctreeVoxelID maxValue, Probability occupancy)
{
  gpu_voxels::Vector3ui coordinates;
  thrust::host_vector<Voxel> voxel(num_points);
  uint32_t max_coordinate = (uint32_t) pow(maxValue, 1.0 / 3);
  for (OctreeVoxelID i = 0; i < num_points; ++i)
  {
    if ((i % (num_points / 10)) == 0)
      printf("Testdata generation progress: %lu%% \n", (OctreeVoxelID) 10 * (i / (num_points / 10)));

    bool isDuplicate = true;
    OctreeVoxelID newID = 0;
    while (isDuplicate)
    {
      //newID = (voxel_id) (drand48() * (maxValue - 1));
      coordinates.x = (uint32_t) (drand48() * (max_coordinate - 1));
      coordinates.y = (uint32_t) (drand48() * (max_coordinate - 1));
      coordinates.z = (uint32_t) (drand48() * (max_coordinate - 1));
      newID = morton_code60(coordinates.x, coordinates.y, coordinates.z);

      isDuplicate = false;
      for (uint32_t k = 0; k < i; k++)
      {
        if (voxel[k].voxelId == newID)
        {
          isDuplicate = true;
          break;
        }
      }
    }

//    printf("vixel_id %lu x: %u y: %u z: %u\n", newID, (uint32_t) coordinates.x, (uint32_t) coordinates.y,
//           (uint32_t) coordinates.z);
    assert(newID < maxValue);
    voxel[i] = Voxel(newID, coordinates, occupancy);
  }
  return voxel;
}

thrust::host_vector<gpu_voxels::Vector3ui> randomCube(
    gpu_voxels::Vector3ui map_dimensions, uint32_t cube_side_length)
{
  gpu_voxels::Vector3ui coordinates;
  thrust::host_vector<gpu_voxels::Vector3ui> h_points(
      cube_side_length * cube_side_length * cube_side_length);

  coordinates.x = (uint32_t) (drand48() * (map_dimensions.x - cube_side_length));
  coordinates.y = (uint32_t) (drand48() * (map_dimensions.y - cube_side_length));
  coordinates.z = (uint32_t) (drand48() * (map_dimensions.z - cube_side_length));

  for (uint32_t xi = 0; xi < cube_side_length; ++xi)
  {
    for (uint32_t yi = 0; yi < cube_side_length; ++yi)
    {
      for (uint32_t zi = 0; zi < cube_side_length; ++zi)
      {
        uint32_t index = xi + yi * cube_side_length + zi * (cube_side_length * cube_side_length);
        h_points[index].x = coordinates.x + xi;
        h_points[index].y = coordinates.y + yi;
        h_points[index].z = coordinates.z + zi;
        assert(h_points[index].x < map_dimensions.x);
        assert(h_points[index].y < map_dimensions.y);
        assert(h_points[index].z < map_dimensions.z);
      }
    }
  }
  return h_points;
}

void translate(thrust::host_vector<gpu_voxels::Vector3ui>& points,
               gpu_voxels::Vector3f translation)
{
  for (uint32_t i = 0; i < points.size(); ++i)
  {
    gpu_voxels::Vector3f tmp = gpu_voxels::Vector3f(points[i].x, points[i].y, points[i].z) + translation;
    points[i] = gpu_voxels::Vector3ui(uint32_t(tmp.x), uint32_t(tmp.y), uint32_t(tmp.z));
  }
}

void rotate(thrust::host_vector<gpu_voxels::Vector3ui>& points, float angle_degree,
            gpu_voxels::Vector3f translation)
{
  float angle_radian = angle_degree / 180.0f * M_PI;

  gpu_voxels::Matrix3f rotation_z = gpu_voxels::Matrix3f::createFromYaw(angle_radian);

  for (uint32_t i = 0; i < points.size(); ++i)
  {
    gpu_voxels::Vector3f tmp = rotation_z
        * gpu_voxels::Vector3f(points[i].x + translation.x, points[i].y + translation.y,
                        points[i].z + translation.z);

    points[i] = gpu_voxels::Vector3ui(uint32_t(tmp.x - translation.x), uint32_t(tmp.y - translation.z),
                               uint32_t(tmp.z - translation.z));
  }
}

thrust::host_vector<OctreeVoxelID> linearVoxel(OctreeVoxelID num_points)
{
  thrust::host_vector<OctreeVoxelID> voxel(num_points);
  for (OctreeVoxelID i = 0; i < num_points; ++i)
    voxel[i] = i;
  return voxel;
}

thrust::host_vector<Voxel> linearVoxel(OctreeVoxelID num_points, OctreeVoxelID offset, Probability occupancy)
{
  gpu_voxels::Vector3ui dummy;
  thrust::host_vector<Voxel> voxel(num_points);
  for (OctreeVoxelID i = 0; i < num_points; ++i)
    voxel[i] = Voxel(offset + i, dummy, occupancy); // TODO: fix coordinates
  return voxel;
}

bool buildTest(std::vector<Vector3f>& points, uint32_t num_points, double & time, bool rebuildTest)
{
  bool error = false;
  typedef NTree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode> NTREE;
//  voxel_id num_points = 128 * 1024 * 1024;
//  rnd.sideLengthInVoxel = 512;
  printf("\n\nbuildTest()\n");

  // std::cout << gpu_voxels::getDeviceMemoryInfo();

//  thrust::host_vector<gpu_voxels::Vector3ui> h_points;
//  // thrust::device_vector<voxel_id> voxel;
//
//  // Allocate memory for points.
//  if (points.empty())
//  {
//    srand(9876);
//    h_points = randomPoints(num_points, NUM_VOXEL);
//
//    std::cout << gpu_voxels::getDeviceMemoryInfo();
//  }
////  for (uint32_t i = 0; i < num_points; ++i)
////    if (voxel[i] == 41)
////      printf("VOXEL 41 FOUND\n");
//
////  for (voxel_id i = 0; i < num_points; ++i)
////    printf("Voxel %i: %lu\n", i, (int64_t) voxel[i]);
//
//  printf("create octree....\n");
//  NTree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode>* o = new NTree<BRANCHING_FACTOR, LEVEL_COUNT,
//      InnerNode, LeafNode>(NUM_BLOCKS, NUM_THREADS_PER_BLOCK);
//
//  gpu_voxels::Vector3ui map_dimensions(0, 0, 0);
//  if (!points.empty())
//  {
//    std::vector<gpu_voxels::Vector3ui> pts;
//    gpu_voxels::Vector3ui center;
//    transformPointCloud(points, pts, map_dimensions, center);
//    h_points = thrust::host_vector<gpu_voxels::Vector3ui>(pts.begin(), pts.end());
//    num_points = pts.size();
//  }
//  else
//  {
//    map_dimensions.x = map_dimensions.y = map_dimensions.z = (uint32_t) ceil(pow(NUM_VOXEL, 1.0 / 3));
//  }
//  printf("num_points: %lu\n", num_points);
//  printf("MapDim x=%u y=%u z=%u\n", map_dimensions.x, map_dimensions.y, map_dimensions.z);
//  printf("Max morton code: %lu Map morton limit: %lu\n",
//         morton_code60(map_dimensions.x - 1, map_dimensions.y - 1, map_dimensions.z - 1),
//         (uint64_t) pow(BRANCHING_FACTOR, LEVEL_COUNT - 1) - 1);
//

  NTREE* o = new NTREE(NUM_BLOCKS, NUM_THREADS_PER_BLOCK);

  BuildResult<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode> build_result;
  if (!buildOctree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode>(o, points, num_points, build_result, 0.1))
    return false;

  //o->build(h_points);
  //o->print2();

  timespec time1 = getCPUTime();
  if (rebuildTest)
  {
    o->rebuild();
  }

  time = timeDiff(time1, getCPUTime());
  printf("build: %f ms\n", time);
  //o->print2();

  printf("AllocInnerNodes: %u    AllocLeafNodes: %u\n", o->allocInnerNodes, o->allocLeafNodes);

//std::cout << "VOXEL CHECK " << voxel[12] << std::endl;

  printf("octree created\n");

//  for (voxel_id i = 0; i < num_points; ++i)
//    printf("Voxel %i: %lu\n", i, (int64_t) hVoxel[i]);

//o->print();

  printf("checking occupied voxel...\n");
  thrust::host_vector<FindResult<LeafNode> > resultNode(num_points);
  o->find(build_result.h_points, resultNode);

// check occupied voxel
  for (OctreeVoxelID i = 0; i < num_points; ++i)
  {
    //printf("Check point %lu voxel %lu\n", i, (voxel_id) hVoxel[i]);
    //std::cout << n.occupationFlags << " " << n.occupationProbablity << std::endl;
    LeafNode n = (LeafNode) resultNode[i].m_node_data;
    if (!n.isOccupied())
    {
      error = true;
      printf("Error! Occupied voxel with x=%u y=%u z=%u, id=%lu not found in octree.\n",
             build_result.h_points[i].x, build_result.h_points[i].y, build_result.h_points[i].z,
             morton_code60(build_result.h_points[i]));
      break;
    }
  }

  printf("checking unknown/free voxel...\n");
// create vector of unknown voxel
  uint32_t max_coordinate = (uint32_t) ceil(pow(NUM_VOXEL, 1.0 / 3));
  printf("max_coordinate: %u\n", max_coordinate);
  {
    thrust::host_vector<gpu_voxels::Vector3ui> unknownVoxel(NUM_VOXEL);
    for (OctreeVoxelID i = 0; i < unknownVoxel.size(); ++i)
    {
      unknownVoxel[i] = gpu_voxels::Vector3ui(
          i % max_coordinate, (i / max_coordinate) % max_coordinate,
          (i / (max_coordinate * max_coordinate)) % max_coordinate);
    }
    for (OctreeVoxelID i = 0; i < num_points; ++i)
      unknownVoxel[build_result.h_points[i].x + build_result.h_points[i].y * max_coordinate
          + build_result.h_points[i].z * (max_coordinate * max_coordinate)] = INVALID_POINT;

    resultNode = thrust::host_vector<FindResult<LeafNode> >(unknownVoxel.size());
    o->find(unknownVoxel, resultNode);

// check unknown voxel
    for (OctreeVoxelID i = 0; i < unknownVoxel.size(); ++i)
    {
      if ((gpu_voxels::Vector3ui)unknownVoxel[i] != INVALID_POINT)
      {
        LeafNode n = (LeafNode) resultNode[i].m_node_data;
        if (n.isOccupied() || n.isFree())
        {
          error = true;
          printf("Error! Unknown voxel with x=%u y=%u z=%u, id=%lu not found in octree.\n", unknownVoxel[i].x,
                 unknownVoxel[i].y, unknownVoxel[i].z, morton_code60(unknownVoxel[i]));
          break;
        }
      }
    }
  }

  error |= o->checkTree();

  delete o;

  if (error)
    printf("##### buildTest() finished with ERRORS #####\n\n\n");
  else
    printf("buildTest() finished\n\n\n");
  return !error;
}

//bool intersectionTest(OctreeVoxelID num_points, enum Intersection_Type insect_type, double & time)
//{
//  printf("Intersection Test ");
//  switch (insect_type)
//  {
//    case SIMPLE:
//      printf("SIMPLE\n");
//      break;
//    case LOAD_BALANCE:
//      printf("LOAD_BALANCE\n");
//      break;
//  }

////  OctreeVoxelID num_points = 1024*1024;
////  rnd.sideLengthInVoxel = 128;

//  std::cout << gpu_voxels::getDeviceMemoryInfo();

//  typedef Environment::InnerNode e_InnerNode;
//  typedef Environment::LeafNode e_LeafNode;
////  typedef Robot::InnerNode r_InnerNode;
////  typedef Robot::LeafNode r_LeafNode;
//  typedef Environment::InnerNode r_InnerNode;
//  typedef Environment::LeafNode r_LeafNode;

//  printf("Environment::LeafNode sizeof(): %lu\n", sizeof(e_LeafNode));
//  printf("Environment::InnerNode sizeof(): %lu\n", sizeof(e_InnerNode));
//  printf("Robot::LeafNode sizeof(): %lu\n", sizeof(r_LeafNode));
//  printf("Robot::InnerNode sizeof(): %lu\n", sizeof(r_InnerNode));

//// way more better results and therefore more intersections
//  srand(12345);
////  thrust::host_vector<OctreeVoxelID> h_r_voxel = randomVoxel(num_points, NUM_VOXEL);
////  thrust::host_vector<OctreeVoxelID> h_e_voxel = randomVoxel(num_points, NUM_VOXEL);
//  thrust::host_vector<OctreeVoxelID> h_r_voxel = linearVoxel(num_points);
//  thrust::host_vector<OctreeVoxelID> h_e_voxel = linearVoxel(num_points);

//// copy to device
//  thrust::device_vector<OctreeVoxelID> r_voxel = h_r_voxel;
//  thrust::device_vector<OctreeVoxelID> e_voxel = h_e_voxel;

////  thrust::sort(h_r_voxel.begin(), h_r_voxel.end());
////  thrust::sort(h_e_voxel.begin(), h_e_voxel.end());

////  for (uint32_t i = 0; i < num_points; ++i)
////    printf("[%i]: Robot %lu    Environment %lu\n", i, (OctreeVoxelID) r_voxel[i], (OctreeVoxelID) e_voxel[i]);

//  gpu_voxels::Vector3ui origin;
//  origin.x = origin.y = origin.z = 0.0f;

//  printf("building robot octree...\n");
//  NTree<BRANCHING_FACTOR, LEVEL_COUNT, r_InnerNode, r_LeafNode>* r_octree = new NTree<BRANCHING_FACTOR,
//      LEVEL_COUNT, r_InnerNode, r_LeafNode>(NUM_BLOCKS, NUM_THREADS_PER_BLOCK);
//  timespec time1 = getCPUTime();
//  r_octree->build(r_voxel, origin);
//  printf("build: %f ms\n", timeDiff(time1, getCPUTime()));
//  printf("AllocInnerNodes: %u    AllocLeafNodes: %u\n", r_octree->allocInnerNodes, r_octree->allocLeafNodes);

////r_octree->print();

//// free voxel
//  r_voxel.clear();
//  r_voxel.shrink_to_fit();

//  printf("robot octree built\n");

//  printf("building environment octree...\n");
//  NTree<BRANCHING_FACTOR, LEVEL_COUNT, e_InnerNode, e_LeafNode>* e_octree = new NTree<BRANCHING_FACTOR,
//      LEVEL_COUNT, e_InnerNode, e_LeafNode>(NUM_BLOCKS, NUM_THREADS_PER_BLOCK);
//  time1 = getCPUTime();
//  e_octree->build(e_voxel, origin);
//  printf("build: %f ms\n", timeDiff(time1, getCPUTime()));
//  printf("AllocInnerNodes: %u    AllocLeafNodes: %u\n", e_octree->allocInnerNodes, e_octree->allocLeafNodes);

////e_octree->print();
//  printf("environment octree built\n");

//// free voxel
//  e_voxel.clear();
//  e_voxel.shrink_to_fit();

//  std::cout << gpu_voxels::getDeviceMemoryInfo();

//// intersect
//  printf("intersecting both...\n");
//  OctreeVoxelID d_numConflicts = 0;

//  cudaProfilerStart();
//  time1 = getCPUTime();
//  switch (insect_type)
//  {
//    case SIMPLE:
//      d_numConflicts = e_octree->intersect(r_octree);
//      break;
//    case LOAD_BALANCE:
//      d_numConflicts = e_octree->intersect_load_balance(r_octree);
//      break;
//  }
//  time = timeDiff(time1, getCPUTime());
//  printf("Intersect: %f ms\n", time);
//  cudaProfilerStop();

//  printf("numConflicts = %lu\n", d_numConflicts);

////return true;

////check numConflicts
//  thrust::sort(h_r_voxel.begin(), h_r_voxel.end());
//  thrust::sort(h_e_voxel.begin(), h_e_voxel.end());
//  thrust::host_vector<OctreeVoxelID> result(num_points);
//  thrust::host_vector<OctreeVoxelID>::iterator end = thrust::set_intersection(h_r_voxel.begin(), h_r_voxel.end(),
//                                                                         h_e_voxel.begin(), h_e_voxel.end(),
//                                                                         result.begin());
//  OctreeVoxelID numEl = end - result.begin();
//  OctreeVoxelID h_numConflicts = (numEl > 0) ? 1 : 0;
//// count manually since there are duplicates
//  for (uint32_t i = 1; i < numEl; ++i)
//    h_numConflicts += ((OctreeVoxelID) result[i - 1] != (OctreeVoxelID) result[i]) ? 1 : 0;
//  printf("real number of conflicts = %lu \n", h_numConflicts);

//  r_octree->free();
//  e_octree->free();

//  if (h_numConflicts == d_numConflicts)
//  {
//    printf("Intersection Test finished\n\n");
//    return true;
//  }
//  else
//  {
//    printf("##### ERROR Intersection Test failed #####\n\n");
//    return false;
//  }
//}

bool insertTest(OctreeVoxelID num_points, OctreeVoxelID num_inserts, bool set_free, bool propergate_up)
{
  typedef NTree<BRANCHING_FACTOR, LEVEL_COUNT, InnerNode, LeafNode> NTREE;

  bool error = false;

  printf("\n\ninsertTest()\n");

// Allocate memory for points.
  srand(TEST_RAND_SEED);
  thrust::host_vector<gpu_voxels::Vector3ui> hVoxel = randomPoints(num_points, NUM_VOXEL); //linearVoxel(num_points);
  CHECK_CUDA_ERROR();

  printf("create octree....\n");
  NTREE* o = new NTREE(NUM_BLOCKS, NUM_THREADS_PER_BLOCK);
  CHECK_CUDA_ERROR();

  timespec time1 = getCPUTime();
  o->build(hVoxel);
  printf("build: %f ms\n", timeDiff(time1, getCPUTime()));
//o->print();

  printf("AllocInnerNodes: %u    AllocLeafNodes: %u\n", o->allocInnerNodes, o->allocLeafNodes);

  printf("octree created\n");

//o->print();

// insert voxel
  thrust::host_vector<Voxel> h_insertVoxel = randomVoxel(num_inserts, NUM_VOXEL, MAX_PROBABILITY);

//  gpu_voxels::Vector3ui t_vec(0, 10, 0);
//  h_insertVoxel[0] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(0, 10, 1);
//  h_insertVoxel[1] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(1, 10, 0);
//  h_insertVoxel[2] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(1, 10, 1);
//  h_insertVoxel[3] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(2, 10, 0);
//  h_insertVoxel[4] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(2, 10, 1);
//  h_insertVoxel[5] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(0, 10, 2);
//  h_insertVoxel[6] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);
//  t_vec = gpu_voxels::Vector3ui(1, 10, 2);
//  h_insertVoxel[7] = Voxel(morton_code60(t_vec), t_vec, MAX_PROBABILITY);

  time1 = getCPUTime();
  thrust::sort(h_insertVoxel.begin(), h_insertVoxel.end()); // TODO: its slow, since its a comparation based sort and not a radix sort; Try to use radix sort instead; e.g. sort_key_value()
  thrust::device_vector<Voxel> d_insertVoxel = h_insertVoxel;
  printf("sort and copy to gpu: %f ms\n", timeDiff(time1, getCPUTime()));

  //  // ##### check if test works #####
  //h_insertVoxel[343].coordinates.x = 132;

  //o->print2();

  printf("insert voxel...\n");
  gpu_voxels::Vector3ui sensor_origin;
  sensor_origin.x = sensor_origin.y = sensor_origin.z = 0;
  time1 = getCPUTime();
  o->insertVoxel(d_insertVoxel, set_free, propergate_up);
  printf("insert: %f ms\n", timeDiff(time1, getCPUTime()));
  printf("voxel inserted\n");

  //o->print2();

  hVoxel.resize(hVoxel.size() + num_inserts);
  for (uint32_t i = 0; i < num_inserts; ++i)
    hVoxel[hVoxel.size() - num_inserts + i] = h_insertVoxel[i].coordinates;

  thrust::host_vector<FindResult<LeafNode> > resultNode(hVoxel.size());
  o->find(hVoxel, resultNode);

  // check occupied voxel of build
  printf("checking occupied voxel of build...\n");
  for (OctreeVoxelID i = 0; i < hVoxel.size() - num_inserts; ++i)
  {
    LeafNode n = (LeafNode) resultNode[i].m_node_data;
    if (!n.isOccupied())
    {
      if (set_free)
      {
        error = true;
        for (uint32_t j = hVoxel.size() - num_inserts; j < hVoxel.size(); ++j)
        {
          if (hVoxel[i] == hVoxel[j])
          {
            error = false;
            break;
          }
        }
      }
      else
        error = true;

      if (error)
      {
        printf("Build Error! Occupied voxel with ID %lu not found in octree.\n",
               (OctreeVoxelID) morton_code60(hVoxel[i]));
        break;
      }
    }
  }

  // check occupied/free voxel of insert
  printf("checking occupied/free voxel of insert...\n");
  for (OctreeVoxelID i = hVoxel.size() - num_inserts; i < hVoxel.size(); ++i)
  {
    LeafNode n = (LeafNode) resultNode[i].m_node_data;
    if ((!set_free && !n.isOccupied()) || (set_free && (!n.isFree() || n.isOccupied())))
    {
      error = true;
      if (set_free)
        printf("Insert Error on Nr. %lu! Free voxel with ID %lu not found in octree.\n",
               i - (hVoxel.size() - num_inserts), (OctreeVoxelID) morton_code60(hVoxel[i]));
      else
        printf("Insert Error on Nr. %lu! Occupied voxel with ID %lu not found in octree.\n",
               i - (hVoxel.size() - num_inserts), (OctreeVoxelID) morton_code60(hVoxel[i]));
      break;
    }
  }

  printf("checking unknown/free voxel...\n");
// create vector of unknown voxel
  thrust::host_vector<gpu_voxels::Vector3ui> unknownVoxel(NUM_VOXEL);
  thrust::host_vector<gpu_voxels::Vector3ui> tmp = linearPoints(NUM_VOXEL, NUM_VOXEL);
  for (OctreeVoxelID i = 0; i < unknownVoxel.size(); ++i)
    unknownVoxel[morton_code60(tmp[i])] = tmp[i];
  for (OctreeVoxelID i = 0; i < hVoxel.size(); ++i)
    unknownVoxel[morton_code60(hVoxel[i])] = INVALID_POINT;

  resultNode = thrust::host_vector<FindResult<LeafNode> >(unknownVoxel.size());
  o->find(unknownVoxel, resultNode);

// check unknown voxel
  for (OctreeVoxelID i = 0; i < unknownVoxel.size(); ++i)
  {
    if (unknownVoxel[i] != INVALID_POINT)
    {
      LeafNode n = resultNode[i].m_node_data;
      if (n.isOccupied() || n.isFree())
      {
        error = true;
        printf("Error! Unknown voxel with ID %lu not found in octree.\n", morton_code60(unknownVoxel[i]));
        break;
      }
    }
  }

  error |= o->checkTree();

  delete o;

  if (error)
    printf("##### insertTest() finished with ERRORS #####\n\n\n");
  else
    printf("insertTest() finished\n\n\n");
  return !error;
}

bool mortonTest(uint32_t num_runs)
{
  //printf("\n\nmortonTest()\n");

  srand(TEST_RAND_SEED);
  srand48(TEST_RAND_SEED);
  const OctreeVoxelID max_voxel_id = (OctreeVoxelID(1) << 60);
  thrust::host_vector<gpu_voxels::Vector3ui> rand_points = randomPoints(num_runs, max_voxel_id);

  for (uint32_t i = 0; i < num_runs; ++i)
  {
    gpu_voxels::Vector3ui coord = (gpu_voxels::Vector3ui) rand_points[i];
    OctreeVoxelID morton_code = morton_code60(coord);
    gpu_voxels::Vector3ui inv_morton_code;
    inv_morton_code60(morton_code, inv_morton_code);
//    printf("coord %u %u %u, morton %lu, inv %u %u %u\n", coord.x, coord.y, coord.z, morton_code,
//           inv_morton_code.x, inv_morton_code.y, inv_morton_code.z);
    if((gpu_voxels::Vector3ui )rand_points[i] != inv_morton_code)
    {
      return false;
    }
  }
  return true;
}

//bool run(std::vector<Vector3f>& points, uint32_t num_points)
//{
//  double time = 0;
//  bool error = false;

//  srand(TEST_RAND_SEED);
//  srand48(TEST_RAND_SEED);


//  Environment::InnerNode n;
//  n.init_h();
//  void* ptr = (void*) 1;
//  n.setChildPtr(ptr);
//  unsigned char* h = (unsigned char*)&n;
//  for (uint i = 0; i < 8; ++i)
//  {
//    printf("%d\n", h[i]);
//  }
//
//  return true;

//  srand(111);
//  while (true)
//  {
//    void* ptr = NULL;
//    voxel_id size = 4 + (voxel_id) (drand48() * 128 * 1024 * 1024);
//    printf("alloc size %lu\n", size);
//    cudaMalloc(&ptr, size);
//    printf("ptr: %p\n", ptr);
//    if (voxel_id(ptr) < 0xB00400000 || voxel_id(ptr) >= 0xC80400000)
//    {
//      printf("ERROR\n");
//      break;
//    }
//  }

//  cudaPointerAttributes attr;
//  void* ptr = NULL;
//  voxel_id size = 2 * 1024;
//  size *= 1024 * 1024;
//  cudaMalloc(&ptr, size);
//  printf("ptr: %p\n", ptr);
////  cudaMalloc(&ptr, 1024 * 1024 * 1024);
////  printf("ptr: %p\n", ptr);
////  cudaMalloc(&ptr, 512 * 1024 * 1024);
////  printf("ptr: %p\n", ptr);
//
//  //void* ptr2 = (void*)0xb00400000;
//  cudaPointerGetAttributes(&attr, ptr);
//  CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st return_data;
//  CUdeviceptr ptr2 = 0;
//  cuMemAlloc_v2(&ptr2, 1024 * 1024);
//  CUresult r = cuPointerGetAttribute(&return_data, CU_POINTER_ATTRIBUTE_P2P_TOKENS, ptr2);
//  printf("blub\n");
//  //CUDA_ERROR_INVALID_VALUE g;
//
//  sleep(10);

//insertTest(54, 1000, time);

//intersectionTest(20 * 64 * 64 * 64, SIMPLE, time);
//  intersectionTest(256 * 256 * 256, LOAD_BALANCE, time);
//intersectionTest( 64 * 64 * 64, LOAD_BALANCE, time);
//intersectionTest(16 * 1024 * 1024, LOAD_BALANCE, time);

//  for (voxel_id i = 136425; i < 1024 * 1024; i += 10)
//  {
//    printf("#####Test Nr. %lu#####\n", i);
//    if (!intersectionTest(i, 105, LOAD_BALANCE))
//    {
//      printf("Error for i = %lu\n", i);
//      break;
//    }
//  }

//  voxel_id num_points = 10;
//  rnd.sideLengthInVoxel = 100;
//  // Allocate memory for points.
//  thrust::device_vector<ulong> voxel(num_points);
//
//  // Generate random points.
//  thrust::generate(voxel.begin(), voxel.end(), rnd);
//
//  gpu_voxels::Vector3ui origin;
//  origin.x = origin.y = origin.z = 0.0f;
//  float voxelSideLength = 1.0f;
//
//  std::cout << "create octree...." << std::endl;
//
//  Octree* o = new Octree(2688, 32);
//  o->build(voxel, origin, rnd.sideLengthInVoxel, voxelSideLength);
//
//  o->print();
//
//  std::cout << "created" << std::endl;

//}

}
}
}
