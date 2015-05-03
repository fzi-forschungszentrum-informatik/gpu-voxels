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
 * \date    2014-02-13
 *
 */
//----------------------------------------------------------------------

#ifndef ARGUMENTHANDLING_H_
#define ARGUMENTHANDLING_H_

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/NTree.h>

#include <stdio.h>
#include <iostream>
#include <vector_types.h>
#include <malloc.h>
#include <vector>

namespace gpu_voxels {
namespace NTree {
namespace Benchmark {

struct Benchmark_Parameter
{
  enum Mode
  {
    MODE_DEFAULT,
    MODE_BUILD_PCF,
    MODE_BUILD_LINEAR,
    MODE_INTERSECT_CUBE,
    MODE_INTERSECT_LINEAR,
    MODE_INTERSECT_PCF,
    MODE_FIND_CUBE,
    MODE_SORT_PCF,
    MODE_SORT_LINEAR,
    MODE_INSERT_ROTATE_PCF,
    MODE_VISUALIZER_PCF,
    MODE_LOAD_PCF,
    MODE_MALLOC
  };

  std::string pc_file;
  std::vector<Vector3f> points;
  uint32_t num_points;
  uint32_t num_runs;
  Mode mode;
  uint32_t robot_cube_side_length_from;
  uint32_t robot_cube_side_length_to;
  std::string command;
  float pcd_scaling;
  uint32_t min_level;
  float kinect_fps;

  Benchmark_Parameter()
  {
    pc_file = "";
    points = std::vector<Vector3f>();
    num_points = 0;
    num_runs = 0;
    mode = MODE_BUILD_PCF;
    robot_cube_side_length_from = 0;
    robot_cube_side_length_to = 0;
    command = "";
    pcd_scaling = 0.0f;
    min_level = 0;
    kinect_fps = 1.0f;
  }

  __host__
    friend std::ostream& operator<<(std::ostream& out, const Benchmark_Parameter& parameter)
  {
    out << "pc_file: " << parameter.pc_file << std::endl << "num_points: " << parameter.num_points
        << std::endl << "num_runs: " << parameter.num_runs << std::endl << "mode: " << parameter.mode
        << std::endl << "robot_cube_side_length_from: " << parameter.robot_cube_side_length_from << std::endl
        << "robot_cube_side_length_to: " << parameter.robot_cube_side_length_to << std::endl << "command: "
        << parameter.command << std::endl << "min_level: " << parameter.min_level << std::endl;
    return out;
  }
};

void parseArguments(Benchmark_Parameter& parameter, int argc, char **argv);

}

namespace Provider {

struct Provider_Parameter
{
  enum Mode
  {
    MODE_NONE, MODE_LOAD_PCF, MODE_KINECT_LIVE, MODE_PTU_LIVE, MODE_KINECT_PLAYBACK, MODE_RANDOM_PLAN, MODE_ROS, MODE_DESERIALIZE
  };

  enum Type
  {
    TYPE_OCTREE, TYPE_VOXEL_MAP, TYPE_OCTOMAP
  };

  enum ModelType
   {
     eMT_Deterministc, eMT_Probabilistic, eMT_BitVector
   };

  std::string pc_file;
  std::string kinect_id;
  std::vector<Vector3f> points;
  uint32_t num_points;
  Mode mode;
  int shared_segment;
  std::string command;
  float kinect_fps;
  bool collide;
  Type type;
  std::size_t max_memory;
  uint32_t resolution_tree;
  uint32_t resolution_occupied;
  uint32_t resolution_free;
  bool free_bounding_box;
  bool swap_x_z;
  int32_t rebuild_frame_count;
  bool manual_kinect;
  bool compute_free_space;
  int sensor_max_range; // in mm
  //bool free_space_packing;
  int num_blocks;
  int num_threads;
  bool deterministic_octree;
  bool save_collisions;
  bool clear_collisions;
  uint32_t min_collision_level;
  Vector3f plan_size; // map dimensions of robot plan in meter
  bool voxelmap_intersect_with_lb;
  Vector3i offset;
  bool serialize;
  ModelType model_type;

  Provider_Parameter()
  {
    pc_file.clear();
    kinect_id.clear();
    points = std::vector<Vector3f>();
    num_points = 0;
    mode = MODE_NONE;
    shared_segment = 0;
    command = "";
    kinect_fps = 1.0f;
    collide = false;
    type = TYPE_OCTREE;
    max_memory = 0;
    resolution_tree = 10;
    resolution_occupied = 10;
    resolution_free = 40;
    free_bounding_box = false;
    swap_x_z = false;
    rebuild_frame_count = -1;
    manual_kinect = false;
    compute_free_space = true;
    sensor_max_range = 5250; // 5250 is max range without artifacts for 1 cm free-space resolution
    //free_space_packing = true;
    num_blocks = 0;
    num_threads = 0;
    deterministic_octree = false;
    save_collisions = true;
    clear_collisions = false;
    min_collision_level = 5;
    plan_size = Vector3f(0);
    voxelmap_intersect_with_lb = false;
    offset = Vector3i(0, 0, 0);
    serialize = false;
    model_type = eMT_Probabilistic;
  }

  __host__
    friend std::ostream& operator<<(std::ostream& out, const Provider_Parameter& parameter)
  {
    return out;
  }

  static void printHelp()
  {
    printf("Parameter:\n");
    printf(
        "Each new run configuration/mode has to start with the 'shm' argument. The other arguments can be in any order.\n");
    printf(
        "   -shm #: (0-2) The number of the shared memory segment to use for communicating with the visualizer.\n");
    printf("   -m #: (load_pc, kinect_live, ptu_live, kinect_playback, ros, deserialize) Mode of the application\n");
    printf("   -fps #: (0-30) Frames per second for a mode which uses the Kinect. Default 1 fps\n");
    printf("   -f #: (file name) File to use for 'load_pc' or 'kinect_playback' mode.\n");
    printf("   -id #: (identifier) Identifer string for the OpenNIGrabber.\n");
    printf(
        "   -resTree #: (1-x) Voxel side length in mm for the smallest voxel of the NTree. Default 10 mm\n");
    printf(
        "   -resOcc #: (1-x) Voxel side length in mm for the smallest voxel for occupied kinect data. Default 10 mm\n");
    printf(
        "   -resFree #: (1-x) Voxel side length in mm for the smallest voxel for free space kinect data. Default 40 mm\n");
    printf("   -c: Collide this data structure with the one of the next 'shm' number.\n");
    printf(
        "   -mem #: (0-x) Max. memory in MB for the VoxelMap, Octree. Default is 0 MB and means no limit.\n");
    printf("   -free: Frees the bounding-box for a map loaded with 'load_pc'\n");
    printf("   -swap_x_z: Swap x and z axis for 'load_pc'\n");
    printf(
        "   -rebuild #: Number of frames before a rebuild of the NTree. Default is -1 and therefore only rebuilds if necessary.\n");
    printf("   -noFreeSpace: Dont't compute the free space. Only use occupied voxel.\n");
    printf(
        "   -maxRange #: Max sensor range. Default 5250 mm. Is max range without artifacts for 1 cm free-space resolution.\n");
    printf("   -det: Use the deterministic Octree. Default is the probabilistic Octree\n");
    printf("   -sx #: Map dimensions of robot plan in meter. Default 10.0 m\n");
    printf("   -sy #: Map dimensions of robot plan in meter. Default 10.0 m\n");
    printf("   -sz #: Map dimensions of robot plan in meter. Default 10.0 m\n");
    printf("   -lb: Use load balancing for NTree and VoxelMap intersection. Default no load balancing\n");
    printf("   -serialize: Write the data into a file on exit of application with STRG+c and $exit\n");
    //printf("   -noFreeSpacePacking: Disables the feature of packing the free space before inserting it into the octree.\n");
    printf("\n\n");
  }
};

bool parseArguments(std::vector<Provider_Parameter>& parameter, int argc, char **argv, bool report_error = true);

bool readPcFile(std::vector<Provider_Parameter>& parameter);

}

namespace Bench {

struct Bech_Parameter
{

  enum Mode
  {
    MODE_NONE, MODE_BUILD, MODE_INSERT, MODE_COLLIDE_LIVE, MODE_COLLIDE
  };

  std::vector<Provider::Provider_Parameter> provider_parameter;
  std::string command;
  Mode mode;
  int runs;
  int resolution_from;
  int resolution_to;
  float resolution_scaling;
  int blocks_from;
  int blocks_to;
  int blocks_step;
  int threads_from;
  int threads_to;
  int threads_step;
  bool log_runs;
  int replay;
  int collision_level_from;
  int collision_level_to;
  int collision_level_step;
  bool save_collisions;

  Bech_Parameter() :
      provider_parameter()
  {
    mode = MODE_NONE;
    runs = 1;
    resolution_from = 10;
    resolution_to = 10;
    resolution_scaling = 2.0;
    blocks_from = NUM_BLOCKS;
    blocks_to = NUM_BLOCKS;
    blocks_step = 1;
    threads_from = NUM_THREADS_PER_BLOCK;
    threads_to = NUM_THREADS_PER_BLOCK;
    threads_step = 1;
    log_runs = false;
    replay = 1;
    collision_level_from = collision_level_to = 0;
    collision_level_step = 1;
    save_collisions = false;
  }

//  __host__
//   friend std::ostream& operator<<(ostream& out, const Provider_Parameter& parameter)
//  {
//    return out;
//  }

//  static void printHelp()
//  {
//    printf("Parameter:\n");
//    printf(
//        "Each new run configuration/mode has to start with the 'shm' argument. The other arguments can be in any order.\n");
//    printf(
//        "   -shm #: (0-2) The number of the shared memory segment to use for communicating with the visualizer.\n");
//    printf("   -m #: (load_pcd, kinect_lice, ptu_live, kinect_playback) Mode of the application\n");
//    printf("   -fps #: (0-30) Frames per second for a mode which uses the Kinect. Default 1 fps\n");
//    printf("   -f #: (file name) File to use for 'load_pcd' or 'kinect_playback' mode.\n");
//    printf("   -resTree #: (1-x) Voxel side length in mm for the smallest voxel of the NTree. Default 10 mm\n");
//    printf("   -resOcc #: (1-x) Voxel side length in mm for the smallest voxel for occupied kinect data. Default 10 mm\n");
//    printf("   -resFree #: (1-x) Voxel side length in mm for the smallest voxel for free space kinect data. Default 10 mm\n");
//    printf("   -c: Collide this data structure with the one of the next 'shm' number.\n");
//    printf(
//        "   -mem #: (0-x) Max. memory in MB of the VoxelMap if it's used. Default is 0 MB and means no limit.\n");
//    printf("   -free: Frees the bounding-box for a map loaded with 'load_pcd'\n");
//    printf("   -swap_x_z: Swap x and z axis for 'load_pcd'\n");
//    printf("   -rebuild #: Number of frames before a rebuild of the NTree. Default is -1 and therefore only rebuilds if necessary.\n");
//    printf("\n\n");
//  }
};

bool parseArguments(Bech_Parameter& parameter, int argc, char **argv, bool report_errors = true);

}

namespace Test {

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
struct BuildResult
{
  double time;
  uint64_t mem_usage;
  uint64_t octree_leaf_nodes;
  uint64_t octree_inner_nodes;
  gpu_voxels::Vector3ui map_dimensions;
  NTree<branching_factor, level_count, InnerNode, LeafNode>* o;
  thrust::host_vector<gpu_voxels::Vector3ui> h_points;
  gpu_voxels::Vector3ui center;
};

bool readPcFile(std::string file_name, std::vector<Vector3f>& points, uint32_t& num_points, bool swap_x_z = false);

}
}
}

#endif /* ARGUMENTHANDLING_H_ */
