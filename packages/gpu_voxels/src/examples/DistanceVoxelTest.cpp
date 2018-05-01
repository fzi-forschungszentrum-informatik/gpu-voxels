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
 * \author  Christian Jülg
 * \date    2015-08-07
 *
 * Display some obstacles, calc distance field, visualize
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#include <gpu_voxels/voxel/DistanceVoxel.h>
#include <gpu_voxels/voxelmap/DistanceVoxelMap.h>

#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>

#include <thrust/remove.h>
#include <string>
#include <sstream>

#ifdef IC_PERFORMANCE_MONITOR
  #include "icl_core_performance_monitor/PerformanceMonitor.h"
  #include <iomanip>
  #include <algorithm>
  #include <vector>
#endif

using boost::dynamic_pointer_cast;
using boost::shared_ptr;
using gpu_voxels::voxelmap::DistanceVoxelMap;

shared_ptr<GpuVoxels> gvl;

const float voxel_side_length = 0.04;
//const float voxel_side_length = 0.02;

//const int NXY = 32;
//const int NXY = 64;

//const int NXY = 128;
//const int NXY = 256;
//const int NXY = 384;
//const int NXY = 480;
const int NXY = 512;

//const int NZ = NXY;
const int NZ = 64;

void ctrlchandler(int)
{
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  exit(EXIT_SUCCESS);
}

bool hollie_cutout(Vector3f p) {
  if (p.x < voxel_side_length*120) return true;
  if (p.x > voxel_side_length*148) return true;
  if (p.y > voxel_side_length*183) return true;
  if (p.y < voxel_side_length*140) return true;
  if (p.z < voxel_side_length*12) return true;
  if (p.z > voxel_side_length*47) return true;
  if (p.y < voxel_side_length*152 && p.x >= voxel_side_length*136 ) return true;
  if (p.y < voxel_side_length*156 && p.z < voxel_side_length*30 ) return true;

  return false;
}

//    Found voxel in voxel map: jfaDistanceVoxmap
//    Voxel position x: 128 y: 161 z: 12
//    Voxel distance x: 128cm y: 161cm z: 12cm
//    Voxel info: 128/161/12

//    Found voxel in voxel map: jfaDistanceVoxmap
//    Voxel position x: 131 y: 143 z: 38
//    Voxel distance x: 131cm y: 143cm z: 38cm
//    Voxel info: 131/143/38

//    Found voxel in voxel map: jfaDistanceVoxmap
//    Voxel position x: 131 y: 143 z: 38
//    Voxel distance x: 131cm y: 143cm z: 38cm
//    Voxel info: 131/143/38

//    Found voxel in voxel map: jfaDistanceVoxmap
//    Voxel position x: 130 y: 163 z: 16
//    Voxel distance x: 130cm y: 163cm z: 16cm
//    Voxel info: 130/163/16

template <uint32_t nxy, uint32_t nz>
struct out_of_bounds {
  bool operator()(Vector3f point) const {
    Vector3ui dimensions(nxy, nxy, nz);

    const Vector3ui int_coords = voxelmap::mapToVoxels(voxel_side_length, point);
//    const Vector3f int_coords = point; //TODO change

    //check if point is in the range of the voxel map
    if ((int_coords.x < dimensions.x) && (int_coords.y < dimensions.y)
        && (int_coords.z < dimensions.z))
    {
      return false;
    } else {
      return true;
    }
  }
};

#ifdef IC_PERFORMANCE_MONITOR
  std::vector<double> get_performance_data(std::string description, std::string prefix)
  {
    return ::icl_core::perf_mon::PerformanceMonitor::getData(description, prefix);
  }
#endif

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize(argc, argv);

  LOGGING_INFO(Gpu_voxels, "DistanceVoxelTest start" << endl);

#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_INITIALIZE(10, 100);
  PERF_MON_ENABLE_ALL(true);
#endif

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */

  const int num_voxels = NXY * NXY * NZ;
  const int distance_voxelmap_bytes = num_voxels * sizeof(DistanceVoxel);

  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(NXY, NXY, NZ, voxel_side_length);
  LOGGING_INFO(Gpu_voxels, "GpuVoxels created, size: " << NXY << "x" << NXY << "x" << NZ << "="<<num_voxels<<" voxel"<< endl);

  //TODO: delay gvl generation to after obstacle loading; make voxel_side_length configurable?; adapt out_of_bounds filtering

  gpu_voxels::DistanceVoxel::accumulated_diff diff_result;

  //TODO: third option: generate given number of points using PRNG and seed?

  // load obstacles
  LOGGING_INFO(Gpu_voxels, "loading obstacles:" << endl);

  std::vector<Vector3f> obstacles;
  Vector3f offset(0.0f);
  float scaling = 1.f;
  std::string pointcloud_filename("<INVALID_FILENAME>");

  LOGGING_INFO(Gpu_voxels, "pointcloud scaling factor: " << scaling << endl);

  int obstacle_model = 8; //valid: 0-8

  if (obstacle_model == 8) {
    pointcloud_filename = "pelican.binvox";
//    scaling = 1.0f;
    scaling = voxel_side_length / 8.0f;
    //    float scaling = 1.0f*(N/128.0f); //scale to voxelmap size
    //    offset = Vector3f(-6, -7.3, -0.2); //ceiling near z=75
    //    offset = Vector3f(-5, -6, -0.2) * scaling;
//    offset = Vector3f(0, -0.4, 0) * scaling;
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }

    for (int i=0; i< 10; i++) {
      LOGGING_INFO(Gpu_voxels, "obstacle " << i << ": "<< obstacles[i].x << "/"<< obstacles[i].y << "/"<< obstacles[i].z << endl);
    }

//    obstacles.erase( std::remove_if(obstacles.begin(), obstacles.end(), hollie_cutout), obstacles.end() );
  }

  if (obstacle_model == 7) {
    pointcloud_filename = "all-decimated-rotated-515z.ply.pcd";
    //    scaling = 1.0f;
        scaling = 16 * voxel_side_length / 1.0f;
    //    float scaling = 1.0f*(N/128.0f); //scale to voxelmap size
    //    offset = Vector3f(-6, -7.3, -0.2); //ceiling near z=75
        //    offset = Vector3f(-5, -6, -0.2) * scaling;
      offset = Vector3f(-30, -210, -20) * (scaling / 8.0f);
//    offset = Vector3f(0, -0.4, 0) * scaling;
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }

    for (int i=0; i< 10; i++) {
      LOGGING_INFO(Gpu_voxels, "obstacle " << i << ": "<< obstacles[i].x << "/"<< obstacles[i].y << "/"<< obstacles[i].z << endl);
    }

//    obstacles.erase( std::remove_if(obstacles.begin(), obstacles.end(), hollie_cutout), obstacles.end() );
  }

  if (obstacle_model == 6) {
    pointcloud_filename = "hollie_from_pointcloud2.pcd";
    scaling = 1.0f;
    //    float scaling = 1.0f*(N/128.0f); //scale to voxelmap size
    //    offset = Vector3f(-6, -7.3, -0.2); //ceiling near z=75
    //    offset = Vector3f(-5, -6, -0.2) * scaling;
//    offset = Vector3f(0, -0.4, 0) * scaling;
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }

//    obstacles.erase( std::remove_if(obstacles.begin(), obstacles.end(), hollie_cutout), obstacles.end() );
  }

  if (obstacle_model == 5) {
    pointcloud_filename = "robot4cmRes.pcd";
    scaling = 1.0f;
    //    float scaling = 1.0f*(N/128.0f); //scale to voxelmap size
    //    offset = Vector3f(-6, -7.3, -0.2); //ceiling near z=75
    //    offset = Vector3f(-5, -6, -0.2) * scaling;
//    offset = Vector3f(0, -0.4, 0) * scaling;
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }

//    obstacles.erase( std::remove_if(obstacles.begin(), obstacles.end(), hollie_cutout), obstacles.end() );
  }

  if (obstacle_model == 3) {

    pointcloud_filename = "pointcloud_0002.pcd";
    scaling = 1.0f;
//    float scaling = 1.0f*(N/128.0f); //scale to voxelmap size
    //    offset = Vector3f(-6, -7.3, -0.2); //ceiling near z=75
//    offset = Vector3f(-5, -6, -0.2) * scaling;
    offset = Vector3f(0, -0.4, 0) * scaling;
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }
  }

  if (obstacle_model == 2) {

    pointcloud_filename = "pointcloud_0002_clipped_rotated.pcd";
    scaling = 1.0f;
//    float scaling = 1.0f*(N/128.0f); //scale to voxelmap size
    //    offset = Vector3f(-6, -7.3, -0.2); //ceiling near z=75
//    offset = Vector3f(-5, -6, -0.2) * scaling;
    offset = Vector3f(0, -0.4, 0) * scaling;
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }
  }

  if (obstacle_model == 1) {

    pointcloud_filename = "hollie/plattform_vereinfacht.binvox";
    scaling = 1.5f*(NXY/128.0f); //at 1.5 platform fits to N=128
    obstacles.clear();

    if(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pointcloud_filename, true, obstacles, true, offset, scaling)) {
      LOGGING_INFO(Gpu_voxels, "loading pointcloud from file "<<pointcloud_filename<<" succeeded!" << endl);
    } else {
      LOGGING_INFO(Gpu_voxels, "using fallback obstacles instead of pointcloud!" << endl);
      obstacle_model = 0;
    }
  }

  if (obstacle_model == 0) {

    obstacles.clear();

    obstacles.push_back(Vector3f(0, 0, 0));
    obstacles.push_back(Vector3f(10, 10, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(60, 60, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(60, 65, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(60, 75, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(60, 80, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(40, 10, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(9, 33, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(9, 30, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(8, 33, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(8, 31, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(7, 32, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(7, 31, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(6, 32, 40) * voxel_side_length);
    obstacles.push_back(Vector3f(6, 30, 40) * voxel_side_length);

    //extra obstacles
    obstacles.push_back(Vector3f(0, 120, 0) * voxel_side_length);
    obstacles.push_back(Vector3f(1, 121, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(0, 121, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(1, 122, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(2, 122, 15) * voxel_side_length);
    obstacles.push_back(Vector3f(3, 123, 16) * voxel_side_length);
    obstacles.push_back(Vector3f(4, 124, 17) * voxel_side_length);
    obstacles.push_back(Vector3f(5, 125, 14) * voxel_side_length);
    //extra obstacles


    //extra obstacles
    obstacles.push_back(Vector3f(21, 112, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(20, 112, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(20, 96, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(22, 127, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(23, 127, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(23, 124, 1) * voxel_side_length);
    //extra obstacles

    //extra obstacles
    obstacles.push_back(Vector3f(70, 0, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(71, 16, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(72, 32, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(73, 48, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(74, 64, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(75, 80, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(76, 96, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(77, 112, 1) * voxel_side_length);
    //extra obstacles

    //extra obstacles
    obstacles.push_back(Vector3f(90, 120, 0) * voxel_side_length);
    obstacles.push_back(Vector3f(91, 121, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(90, 121, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(91, 122, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(92, 122, 15) * voxel_side_length);
    obstacles.push_back(Vector3f(93, 123, 16) * voxel_side_length);
    obstacles.push_back(Vector3f(94, 124, 17) * voxel_side_length);
    obstacles.push_back(Vector3f(95, 125, 14) * voxel_side_length);
    //extra obstacles

    //extra obstacles
    obstacles.push_back(Vector3f(90, 0, 0) * voxel_side_length);
    obstacles.push_back(Vector3f(91, 1, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(90, 1, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(91, 2, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(92, 2, 15) * voxel_side_length);
    obstacles.push_back(Vector3f(93, 3, 16) * voxel_side_length);
    obstacles.push_back(Vector3f(94, 4, 17) * voxel_side_length);
    obstacles.push_back(Vector3f(95, 5, 14) * voxel_side_length);
    //extra obstacles

    //extra obstacles
    obstacles.push_back(Vector3f(0, 1, 0) * voxel_side_length);
    obstacles.push_back(Vector3f(1, 1, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(0, 1, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(1, 2, 1) * voxel_side_length);
    obstacles.push_back(Vector3f(2, 2, 15) * voxel_side_length);
    obstacles.push_back(Vector3f(3, 3, 16) * voxel_side_length);
    obstacles.push_back(Vector3f(4, 4, 17) * voxel_side_length);
    obstacles.push_back(Vector3f(5, 5, 14) * voxel_side_length);
    //extra obstacles
  }

  if (obstacle_model == 4) {

    obstacles.clear();

    int z = NZ / 2;

    //TODO: use sparse points to paint room with door -> box with part of one side missing

    int square_length = 9;
    for (int i=0; i<square_length; i++) {
      for (int j=0; j<square_length; j++) {
        if (   (i == 0)
            || (j == 0)
            || (i == (square_length -1))
            || (j == (square_length -1))
            ) {
          if (!(i == 0 && j > 2 && j < 6)) {
            int spread_factor = 3;
            obstacles.push_back(Vector3f(((NXY/2) + i*spread_factor)*voxel_side_length, ((NXY/2) + j*spread_factor)*voxel_side_length, z*voxel_side_length));
          }
        }
      }
    }
  }

  LOGGING_INFO(Gpu_voxels, "obstacle count before filtering: "<<obstacles.size()<< endl);

  std::vector<Vector3f>::iterator new_end = thrust::remove_if(obstacles.begin(), obstacles.end(), out_of_bounds<NXY, NZ>());
  obstacles.erase(new_end, obstacles.end());
  //    obstacles.resize(new_end - obstacles.begin());

  LOGGING_INFO(Gpu_voxels, "obstacle count after bounds filtering: "<< obstacles.size() << endl);

  {
    std::vector<bool> voxels_found(num_voxels, false);
    int num_found = 0;
    std::vector<Vector3f> unique_obstacles;
    for (uint obstacle_i=0; obstacle_i<obstacles.size(); obstacle_i++) {
      Vector3f obstacle = obstacles[obstacle_i];
      Vector3ui int_coords = voxelmap::mapToVoxels(voxel_side_length, obstacle);
      int linear_id = (int_coords.z * NXY  * NXY) + (int_coords.y * NXY) + int_coords.x;
      if (voxels_found[linear_id] == false) {
        voxels_found[linear_id] = true;
        num_found++;
        unique_obstacles.push_back(obstacle);
      }
    }
    //      LOGGING_INFO(Gpu_voxels, "num_found "<< num_found << ", unique_obstacles size: " << unique_obstacles.size() << endl);
    obstacles = unique_obstacles;
  }

  LOGGING_INFO(Gpu_voxels, "obstacle count after voxel duplicate filtering: "<< obstacles.size() << endl);

//  bool create_exactDistanceMap = obstacles.size() < 10 * 1000; //diff computation would take a long time
    bool create_exactDistanceMap = false; //diff computation would take a long time
//    bool create_exactDistanceMap = true; //diff computation would take a long time

  bool create_emptyMaps = false; //only used to visualize PBA inner workings
//  bool create_emptyMaps = true; //only used to visualize PBA inner workings

  const int JFA_RUNS = 1;
  const int PBA_RUNS = 11;

  int show_detailed_timing = 0;

#ifdef IC_PERFORMANCE_MONITOR
  std::map<std::string, double> median_timings;
  std::map<std::string, double> min_timings;
  std::map<std::string, double> total_timings;
#endif

//  LOGGING_INFO(Gpu_voxels, "    ================  2D distances  ================    " << endl);

//  gvl->addMap(MT_DISTANCE_VOXELMAP, "emptyDistanceVoxmap");
//  shared_ptr<DistanceVoxelMap> emptyDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("emptyDistanceVoxmap"));
//  emptyDistanceVoxmap->clearMap();

//  gvl->addMap(MT_DISTANCE_VOXELMAP, "emptyTransformedDistanceVoxmap");
//  shared_ptr<DistanceVoxelMap> emptyTransformedDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("emptyTransformedDistanceVoxmap"));
//  emptyTransformedDistanceVoxmap->clearMap();
//  emptyTransformedDistanceVoxmap->pba_transform();

//  // calculate exact distances
//  exactDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
//  LOGGING_INFO(Gpu_voxels, "exactDistances 2D start" << endl);
//  exactDistanceVoxmap->exactDistances2D(40);
//  LOGGING_INFO(Gpu_voxels, "exactDistances 2D done" << endl);

//  // calculate distances using JFA
//  jfaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
//  LOGGING_INFO(Gpu_voxels, "jumpflood of DistanceVoxelMap starting with buffer" << endl);
//  jfaDistanceVoxmap->jumpFlood2D(40, true);
//  LOGGING_INFO(Gpu_voxels, "jumpflood of DistanceVoxelMap complete" << endl);
//  // evaluate differences. output result struct
//  LOGGING_INFO(Gpu_voxels, "compare exact2DDistances and jumpFlood2D voxels with buffer: " << endl);
//  gpu_voxels::DistanceVoxel::accumulated_diff diff_result = exactDistanceVoxmap->differences3D(jfaDistanceVoxmap);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

//  //TODO remove unbuffered run
//  //again without double buffer
//  jfaDistanceVoxmap->clearMap();
//  jfaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
//  LOGGING_INFO(Gpu_voxels, "jumpflood of DistanceVoxelMap starting without buffer" << endl);
//  jfaDistanceVoxmap->jumpFlood2D(40, false);
//  LOGGING_INFO(Gpu_voxels, "jumpflood of DistanceVoxelMap complete" << endl);
//  // evaluate differences. output result struct
//  LOGGING_INFO(Gpu_voxels, "compare exact2DDistances and jumpFlood2D voxels without buffer: " << endl);
//  diff_result = exactDistanceVoxmap->differences3D(jfaDistanceVoxmap);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

//  // compare jumpFlood to empty map
//  LOGGING_INFO(Gpu_voxels, "compare empty voxels and jumpFlood2D voxels: " << endl);
//  diff_result = emptyDistanceVoxmap->differences3D(jfaDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

  //TODO: create emptyTransformed 2D map?
//  LOGGING_INFO(Gpu_voxels, "compare emptyTransformed voxels and jumpFlood2D voxels: " << endl);
//  diff_result = emptyTransformedDistanceVoxmap->differences3D(jfaDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

  //3D methods

  LOGGING_INFO(Gpu_voxels, "    ================  3D distances  ================    " << endl);

  gvl->addMap(MT_DISTANCE_VOXELMAP, "jfaDistanceVoxmap");
  shared_ptr<DistanceVoxelMap> jfaDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("jfaDistanceVoxmap"));

  // calculate distances using JFA
  int jfa_block_size = cMAX_THREADS_PER_BLOCK;

#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_INITIALIZE(10, 100);
#endif

  LOGGING_INFO(Gpu_voxels, "jumpFlood 3D start" << endl);
#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_START("outsidetimer");
#endif
  //3D JFA
  jfaDistanceVoxmap->clearMap();
#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_PRINT_INFO_P("outsidetimer", "jfa clearMap done", "outsideprefix");
#endif

  jfaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
#ifdef IC_PERFORMANCE_MONITOR
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  PERF_MON_PRINT_INFO_P("outsidetimer", "jfa clearMap and insertPointloud done", "outsideprefix");
#endif

  // calculate distances using JFA
  jfaDistanceVoxmap->jumpFlood3D(jfa_block_size, 1, false);
#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_PRINT_AND_RESET_INFO_P("outsidetimer", "jfa done", "outsideprefix");
#endif
  LOGGING_INFO(Gpu_voxels, "jumpFlood 3D done" << endl);


  LOGGING_INFO(Gpu_voxels, "starting " << JFA_RUNS << " JFA runs; " << endl);

#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_INITIALIZE(10, 100);
#endif

  //TODO: run repeatedly, collect min/median runtime!
  //TODO: for jfa_block_size

  for (int run_id = 0; run_id < JFA_RUNS; run_id++) {

    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef IC_PERFORMANCE_MONITOR
    PERF_MON_START("outsidetimer");
#endif
    jfaDistanceVoxmap->clearMap();
    jfaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
#ifdef IC_PERFORMANCE_MONITOR
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    PERF_MON_START("computetimer");
#endif
  jfaDistanceVoxmap->jumpFlood3D(jfa_block_size, 0, false);
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
#ifdef IC_PERFORMANCE_MONITOR
    PERF_MON_PRINT_AND_RESET_INFO_P("computetimer", "jfa compute done", "computeprefix");
    PERF_MON_PRINT_AND_RESET_INFO_P("outsidetimer", "jfa total done", "outsideprefix");
#endif
  }

#ifdef IC_PERFORMANCE_MONITOR
  std::vector<double> compute_times = get_performance_data("jfa compute done", "computeprefix");
  std::sort(compute_times.begin(), compute_times.end());
  double median_computetime = compute_times[compute_times.size() / 2];
  double min_computetime = compute_times[0];
  std::stringstream ss;

  ss << jfa_block_size << "; ";

  median_timings[ss.str()] = median_computetime;
  min_timings[ss.str()] = min_computetime;

  std::vector<double> total_times = get_performance_data("jfa total done", "outsideprefix");
  std::sort(total_times.begin(), total_times.end());
  double median_totaltime = total_times[total_times.size() / 2];
  total_timings[ss.str()] = median_totaltime;
#endif

#ifdef IC_PERFORMANCE_MONITOR
  // print timings summary

  //headline:
  std::string title("jfa_block_size; compute_time_ms; num_voxels; NX; NY; NZ; num_obstacles; compute_voxel_throughput; compute DV throughput byte/s; compute_framerate; sizeof(DistanceVoxel);");

  LOGGING_INFO(Gpu_voxels, "collected " << median_timings.size() << " median compute times" << endl);
  LOGGING_INFO(Gpu_voxels, title << endl);
  for (std::map<std::string, double>::iterator it = median_timings.begin(); it != median_timings.end(); it++) {
    std::stringstream ss;
    ss << "" << it->first << std::setprecision(5) << std::setfill('0') << it->second << ";";
    ss << num_voxels << ";" << NXY << ";" << NXY << ";" << NZ << ";" << obstacles.size() << ";";
    ss << std::setprecision(12) << (num_voxels/(it->second/1000)) << ";";
    ss << std::setprecision(12) << (distance_voxelmap_bytes/(it->second/1000)) << ";";
    ss << std::setprecision(6) << (1/(it->second/1000)) << ";";
    ss << sizeof(DistanceVoxel) << ";";
    LOGGING_INFO(Gpu_voxels, ss.str() << endl);
  }

  LOGGING_INFO(Gpu_voxels, "collected " << min_timings.size() << " minimum compute times" << endl);
  LOGGING_INFO(Gpu_voxels, title << endl);
  for (std::map<std::string, double>::iterator it = min_timings.begin(); it != min_timings.end(); it++) {
    std::stringstream ss;
    ss << "" << it->first << std::setprecision(5) << std::setfill('0') << it->second << ";";
    ss << num_voxels << ";" << NXY << ";" << NXY << ";" << NZ << ";" << obstacles.size() << ";";
    ss << std::setprecision(12) << (num_voxels/(it->second/1000)) << ";";
    ss << std::setprecision(12) << (distance_voxelmap_bytes/(it->second/1000)) << ";";
    ss << std::setprecision(6) << (1/(it->second/1000)) << ";";
    ss << sizeof(DistanceVoxel) << ";";
    LOGGING_INFO(Gpu_voxels, ss.str() << endl);
  }

  title = std::string("jfa_block_size; total_time_ms; num_voxels; NX; NY; NZ; num_obstacles; total_voxel_throughput; total DV throughput byte/s; total_framerate; sizeof(DistanceVoxel);");
  LOGGING_INFO(Gpu_voxels, "collected " <<total_timings.size() << " total times (clear + insert + init + compute)" << endl);
  LOGGING_INFO(Gpu_voxels, title << endl);
  for (std::map<std::string, double>::iterator it = total_timings.begin(); it != total_timings.end(); it++) {
    std::stringstream ss;
    ss << "" << it->first << std::setprecision(5) << std::setfill('0') << it->second << ";";
    ss << num_voxels << ";" << NXY << ";" << NXY << ";" << NZ << ";" << obstacles.size() << ";";
    ss << std::setprecision(12) << (num_voxels/(it->second/1000)) << ";";
    ss << std::setprecision(12) << (distance_voxelmap_bytes/(it->second/1000)) << ";";
    ss << std::setprecision(6) << (1/(it->second/1000)) << ";";
    ss << sizeof(DistanceVoxel) << ";";
    LOGGING_INFO(Gpu_voxels, ss.str() << endl);
  }
#endif

#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_SUMMARY_ALL_INFO;
#endif

  //PBA
  gvl->addMap(MT_DISTANCE_VOXELMAP, "pbaDistanceVoxmap");
  shared_ptr<DistanceVoxelMap> pbaDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("pbaDistanceVoxmap"));

#ifdef IC_PERFORMANCE_MONITOR
  median_timings.clear();
  min_timings.clear();
  total_timings.clear();
#endif

  //TODO DELETE: canary
  pbaDistanceVoxmap->clearMap();
  pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
  pbaDistanceVoxmap->parallelBanding3D(1, 4, 1, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);
  //TODO DELETE


  //check 1, 2, 4, 8, 16?
  int factor = 2;
  uint32_t parameter_min = 1;
  uint32_t parameter_max = 4;
  //    uint32_t parameter_max = 32;

  // test various vals, print timings after each
  for (uint32_t parameter_choice = 1; parameter_choice <= 3; parameter_choice++)
  {

    for (uint32_t parameter_val = parameter_min; parameter_val <= parameter_max; parameter_val *= factor)
    {
      uint32_t m1 = 1, m2 = 1, m3 = 1;
      switch (parameter_choice) {
      case 1:
        m1 = parameter_val;
        break;
      case 2:
        m2 = parameter_val;
        break;
      case 3:
        m3 = parameter_val;
        break;
      default: LOGGING_INFO(Gpu_voxels, "parameter_choice invalid! " << parameter_choice << endl); break;
      }

      if (m1+m2+m3 == 3 && parameter_choice != 1) continue; //test the m1=m2=m3=1 case only once

      for (float blocksize_factor = 1; blocksize_factor <= 1; blocksize_factor *= 2)
      {
        uint32_t m1_block_size = PBA_DEFAULT_M1_BLOCK_SIZE;
        uint32_t m2_block_size = PBA_DEFAULT_M2_BLOCK_SIZE;
        uint32_t m3_block_size = PBA_DEFAULT_M3_BLOCK_SIZE;
        std::string blocksize_description;
        {
          std::stringstream ss;
          switch (parameter_choice) {
          case 1:
            m1_block_size = PBA_DEFAULT_M1_BLOCK_SIZE * blocksize_factor;
            ss << "m1 BS = " << m1_block_size;
            break;
          case 2:
            m2_block_size = PBA_DEFAULT_M2_BLOCK_SIZE * blocksize_factor;
            ss << "m2 BS = " << m2_block_size;
            break;
          case 3:
            m3_block_size = PBA_DEFAULT_M3_BLOCK_SIZE * blocksize_factor;
            if (blocksize_factor > 1) { // phase3 uses a lot of shared memory? blocksize >= 32 leads to illegal memory access
//              LOGGING_INFO(Gpu_voxels, "unsafe combination of m3 (" << m3 << ") and m3_blocksize (" << m3_block_size << ")" << endl);
              continue;
            }
            ss << "m3 BS = " << m3_block_size;
            break;
          default: LOGGING_INFO(Gpu_voxels, "parameter_choice invalid! " << parameter_choice << endl); break;
          }
          blocksize_description = ss.str();
        }

        LOGGING_INFO(Gpu_voxels, "starting " << PBA_RUNS << " PBA runs; " << blocksize_description << endl);

#ifdef IC_PERFORMANCE_MONITOR
        PERF_MON_INITIALIZE(10, 100);
#endif

        for (int run_id = 0; run_id < PBA_RUNS; run_id++) {

#ifdef IC_PERFORMANCE_MONITOR
          PERF_MON_START("outsidetimer");
#endif

          pbaDistanceVoxmap->clearMap();
#ifdef IC_PERFORMANCE_MONITOR
          //      PERF_MON_PRINT_INFO_P("outsidetimer", "pba iteration clearMap done", "outsideprefix");
#endif

          pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
#ifdef IC_PERFORMANCE_MONITOR
          //      HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
          //      PERF_MON_PRINT_INFO_P("outsidetimer", "pba iteration clearMap and insertPointloud done", "outsideprefix");
#endif

          //      LOGGING_INFO(Gpu_voxels, "parallelBanding of DistanceVoxelMap starting, obstacle list has " << obstacles.size() << " elements." << endl);
          pbaDistanceVoxmap->parallelBanding3D(m1, m2, m3, m1_block_size, m2_block_size, m3_block_size, show_detailed_timing);
          //      LOGGING_INFO(Gpu_voxels, "parallelBanding of DistanceVoxelMap done" << endl);

#ifdef IC_PERFORMANCE_MONITOR
          PERF_MON_PRINT_AND_RESET_INFO_P("outsidetimer", "pba iteration done", "outsideprefix");
#endif
        }

#ifdef IC_PERFORMANCE_MONITOR
        std::vector<double> compute_times = get_performance_data("parallelBanding3D compute done", "pbaprefix");
        std::sort(compute_times.begin(), compute_times.end());
        double median_computetime = compute_times[compute_times.size() / 2];
        double min_computetime = compute_times[0];
        std::stringstream ss;

        ss << std::setw(2) << std::setfill('0') << m1 << ";"
           << std::setw(2) << std::setfill('0') << m2 << ";"
           << std::setw(2) << std::setfill('0') << m3 << ";"
           << std::setw(3) << std::setfill('0') << m1_block_size << ";"
           << std::setw(3) << std::setfill('0') << m2_block_size << ";"
           << std::setw(3) << std::setfill('0') << m3_block_size << "; ";

        median_timings[ss.str()] = median_computetime;
        min_timings[ss.str()] = min_computetime;

        std::vector<double> total_times = get_performance_data("pba iteration done", "outsideprefix");
        std::sort(total_times.begin(), total_times.end());
        double median_totaltime = total_times[total_times.size() / 2];
        total_timings[ss.str()] = median_totaltime;
#endif

#ifdef IC_PERFORMANCE_MONITOR
        //    PERF_MON_SUMMARY_PREFIX_INFO("outsideprefix");
        //    PERF_MON_SUMMARY_PREFIX_INFO("pbaprefix");
        //    if (m1 == 1 && m2 == 1 && m3 == 1) PERF_MON_SUMMARY_PREFIX_INFO("");
        ////      PERF_MON_SUMMARY_ALL_INFO;
#endif

      }
    }
  }

#ifdef IC_PERFORMANCE_MONITOR
  // print timings summary

  //headline:
  title = std::string("m1 ; m2; m3; m1_bs; m2_bs; m3_bs; compute_time_ms; num_voxels; NX; NY; NZ; num_obstacles; compute_voxel_throughput; compute DV throughput byte/s; compute_framerate; sizeof(pba_fw_trt_t); sizeof(DistanceVoxel);");

  LOGGING_INFO(Gpu_voxels, "collected " << median_timings.size() << " median compute times" << endl);
  LOGGING_INFO(Gpu_voxels, title << endl);
  for (std::map<std::string, double>::iterator it = median_timings.begin(); it != median_timings.end(); it++) {
    std::stringstream ss;
    ss << "" << it->first << std::setprecision(5) << std::setfill('0') << it->second << ";";
    ss << num_voxels << ";" << NXY << ";" << NXY << ";" << NZ << ";" << obstacles.size() << ";";
    ss << std::setprecision(12) << (num_voxels/(it->second/1000)) << ";";
    ss << std::setprecision(12) << (distance_voxelmap_bytes/(it->second/1000)) << ";";
    ss << std::setprecision(6) << (1/(it->second/1000)) << ";";
    ss << sizeof(gpu_voxels::pba_fw_ptr_t) << ";" << sizeof(DistanceVoxel) << ";";
    LOGGING_INFO(Gpu_voxels, ss.str() << endl);
  }

  LOGGING_INFO(Gpu_voxels, "collected " << min_timings.size() << " minimum compute times" << endl);
  LOGGING_INFO(Gpu_voxels, title << endl);
  for (std::map<std::string, double>::iterator it = min_timings.begin(); it != min_timings.end(); it++) {
    std::stringstream ss;
    ss << "" << it->first << std::setprecision(5) << std::setfill('0') << it->second << ";";
    ss << num_voxels << ";" << NXY << ";" << NXY << ";" << NZ << ";" << obstacles.size() << ";";
    ss << std::setprecision(12) << (num_voxels/(it->second/1000)) << ";";
    ss << std::setprecision(12) << (distance_voxelmap_bytes/(it->second/1000)) << ";";
    ss << std::setprecision(6) << (1/(it->second/1000)) << ";";
    ss << sizeof(gpu_voxels::pba_fw_ptr_t) << ";" << sizeof(DistanceVoxel) << ";";
    LOGGING_INFO(Gpu_voxels, ss.str() << endl);
  }

  title = std::string("m1 ; m2; m3; m1_bs; m2_bs; m3_bs; total_time_ms; num_voxels; NX; NY; NZ; num_obstacles; total_voxel_throughput; total DV throughput byte/s; total_framerate; sizeof(pba_fw_trt_t); sizeof(DistanceVoxel);");
  LOGGING_INFO(Gpu_voxels, "collected " <<total_timings.size() << " total times (clear + insert + init + compute)" << endl);
  LOGGING_INFO(Gpu_voxels, title << endl);
  for (std::map<std::string, double>::iterator it = total_timings.begin(); it != total_timings.end(); it++) {
    std::stringstream ss;
    ss << "" << it->first << std::setprecision(5) << std::setfill('0') << it->second << ";";
    ss << num_voxels << ";" << NXY << ";" << NXY << ";" << NZ << ";" << obstacles.size() << ";";
    ss << std::setprecision(12) << (num_voxels/(it->second/1000)) << ";";
    ss << std::setprecision(12) << (distance_voxelmap_bytes/(it->second/1000)) << ";";
    ss << std::setprecision(6) << (1/(it->second/1000)) << ";";
    ss << sizeof(gpu_voxels::pba_fw_ptr_t) << ";" << sizeof(DistanceVoxel) << ";";
    LOGGING_INFO(Gpu_voxels, ss.str() << endl);
  }
#endif

  //TODO: check PBA results against each other? use "sync always" as reference

  if (create_exactDistanceMap)
  {
    LOGGING_INFO(Gpu_voxels, "    ================  3D distances compared to exact distances  ================    " << endl);

    gvl->addMap(MT_DISTANCE_VOXELMAP, "exactDistanceVoxmap");
    shared_ptr<DistanceVoxelMap> exactDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("exactDistanceVoxmap"));

    //compute 3D exact distances
    exactDistanceVoxmap->clearMap();
    exactDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    LOGGING_INFO(Gpu_voxels, "exactDistances 3D start" << endl);
    exactDistanceVoxmap->exactDistances3D(obstacles);
    LOGGING_INFO(Gpu_voxels, "exactDistances 3D done" << endl);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare exact3DDistances and JFA 3D voxels: " << endl);
    diff_result = exactDistanceVoxmap->differences3D(jfaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare PBA and JFA 3D voxels: " << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare exact3DDistances and PBA 3D voxels: " << endl);
    diff_result = exactDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(1, 1, 1, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare exact3DDistances and PBA 3D voxels: (m1,2,3 = 1)" << endl);
    diff_result = exactDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);


    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(4, 4, 4, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare exact3DDistances and PBA 3D voxels: (m1,2,3 = 4)" << endl);
    diff_result = exactDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

  } else {

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, "compare jfaDistanceVoxmap and PBA 3D voxels: " << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, 0, false); //no debug output, dont reinit PerfMon
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(1, 1, 1, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare jfaDistanceVoxmap and PBA 3D voxels: (m1,2,3 = 1)" << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);


    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(4, 1, 4, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare jfaDistanceVoxmap and PBA 3D voxels: (m1=4,m2=1,m3=4)" << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);


    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(1, 4, 4, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare jfaDistanceVoxmap and PBA 3D voxels: (m1=1,m2=4,m3=4)" << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);


    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(4, 4, 4, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare jfaDistanceVoxmap and PBA 3D voxels: (m1,2,3 = 4)" << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

    pbaDistanceVoxmap->clearMap();
    pbaDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
    pbaDistanceVoxmap->parallelBanding3D(10, 16, 4, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    // evaluate differences. output result struct
    LOGGING_INFO(Gpu_voxels, endl << "compare jfaDistanceVoxmap and PBA 3D voxels: (m1,2,3 = 4)" << endl);
    diff_result = jfaDistanceVoxmap->differences3D(pbaDistanceVoxmap, false);
    LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);
}

//  //empty comparisons
//  LOGGING_INFO(Gpu_voxels, "    ================  3D distances compared to empty maps ================    " << endl);

//  // compare exactDistances3D to empty map
//  LOGGING_INFO(Gpu_voxels, "compare exactDistances3D voxels and empty voxels: " << endl);
//  diff_result = emptyDistanceVoxmap->differences3D(exactDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

//  // compare jumpFlood 3D to empty map
//  LOGGING_INFO(Gpu_voxels, "compare empty voxels and jumpFlood3D voxels: " << endl);
//  diff_result = emptyDistanceVoxmap->differences3D(jfaDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

//  LOGGING_INFO(Gpu_voxels, "compare emptyTransformed voxels and jumpFlood3D voxels: " << endl);
//  diff_result = emptyTransformedDistanceVoxmap->differences3D(jfaDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

//  // compare PBA 3D to empty map
//  LOGGING_INFO(Gpu_voxels, "compare empty voxels and PBA 3D voxels: " << endl);
//  diff_result = emptyDistanceVoxmap->differences3D(pbaDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

//  LOGGING_INFO(Gpu_voxels, "compare emptyTransformed voxels and PBA 3D voxels: " << endl);
//  diff_result = emptyTransformedDistanceVoxmap->differences3D(pbaDistanceVoxmap, 1);
//  LOGGING_INFO(Gpu_voxels, diff_result.str() << endl);

  if (create_emptyMaps) {
    LOGGING_INFO(Gpu_voxels, "adding emptyDistanceVoxmap" << endl);
    gvl->addMap(MT_DISTANCE_VOXELMAP, "emptyDistanceVoxmap");

    LOGGING_INFO(Gpu_voxels, "adding emptyClearedDistanceVoxmap" << endl);
    gvl->addMap(MT_DISTANCE_VOXELMAP, "emptyClearedDistanceVoxmap");
    shared_ptr<DistanceVoxelMap> emptyClearedDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("emptyClearedDistanceVoxmap"));
    emptyClearedDistanceVoxmap->clearMap();

    LOGGING_INFO(Gpu_voxels, "adding emptyClearInsertDistanceVoxmap" << endl);
    gvl->addMap(MT_DISTANCE_VOXELMAP, "emptyClearInsertDistanceVoxmap");
    shared_ptr<DistanceVoxelMap> emptyClearInsertDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("emptyClearInsertDistanceVoxmap"));
    emptyClearInsertDistanceVoxmap->clearMap();
    emptyClearInsertDistanceVoxmap->insertPointCloud(obstacles, eBVM_OCCUPIED);
  }


  LOGGING_INFO(Gpu_voxels, "start visualizing maps" << endl);

  //TODO: measure timing, print. use icl_? or boost_? loop process N times, get average/median results? consider nondeterministic JFA results

  // tell the visualier wich maps should be shown
  if (create_exactDistanceMap) gvl->visualizeMap("exactDistanceVoxmap");
  gvl->visualizeMap("jfaDistanceVoxmap");
  gvl->visualizeMap("pbaDistanceVoxmap");
  if (create_emptyMaps) {
    //TODO delete:
    gvl->visualizeMap("emptyDistanceVoxmap");
    gvl->visualizeMap("emptyClearedDistanceVoxmap");
    gvl->visualizeMap("emptyClearInsertDistanceVoxmap");
  }

  if (argc == 1) {
    std::cout.flush();
    LOGGING_INFO(Gpu_voxels, "press enter to exit" << endl);
    std::string in;
    std::getline(std::cin, in);
  } else {
//      for (int i=0; i<argc; i++) {
//          std::cout << "arg[" << i << "]: " << argv[i] << std::endl;
//      }
  }

  LOGGING_INFO(Gpu_voxels, "shutting down" << endl);

  // remove the maps:
  gvl->delMap("jfaDistanceVoxmap");
  gvl->delMap("pbaDistanceVoxmap");
  if (create_exactDistanceMap) gvl->delMap("exactDistanceVoxmap");

  if (create_emptyMaps) {
    gvl->delMap("emptyDistanceVoxmap");
    gvl->delMap("emptyClearedDistanceVoxmap");
    gvl->delMap("emptyClearInsertDistanceVoxmap");
  }

  exit(EXIT_SUCCESS);
}
