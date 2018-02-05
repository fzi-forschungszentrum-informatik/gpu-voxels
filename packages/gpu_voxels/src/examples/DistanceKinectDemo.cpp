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
 * \author  Andreas Hermann
 * \date    2016-12-24
 *
 * This demo calcuates a distance field on the pointcloud
 * captured by a Kinect camera.
 * Two virtual meausrement points are places in the scene
 * from which the clearance to their closest Kinect obstacle
 * is constantly measured (and printed on terminal).
 *
 * Place the Kinect so it faces you in a distance of about
 * 1.5 meters.
 *
 * Start the demo and then the visualizer.
 * Press "ALT-t" two times, then "s" two times.
 * You will see the Kinect pointcloud inflated by 10 Voxels.
 * Use "t" to switch through 3 slicing modes and "a" or "q"
 * to move the slice.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>

#include <gpu_voxels/helpers/Kinect.h>
#include <icl_core_config/Config.h>

using boost::dynamic_pointer_cast;
using boost::shared_ptr;
using gpu_voxels::voxelmap::DistanceVoxelMap;

shared_ptr<GpuVoxels> gvl;

Vector3ui map_dimensions(256, 256, 256);
const float voxel_side_length = 0.01;

// Define Parallel Banding parameters:
uint32_t m1 = 1;
uint32_t m2 = 1;
uint32_t m3 = 4;

void ctrlchandler(int)
{
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  exit(EXIT_SUCCESS);
}


int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::config::GetoptParameter ident_parameter("device-identifier:", "id",
                                                    "Identifer of the kinect device");
  icl_core::config::addParameter(ident_parameter);
  icl_core::logging::initialize(argc, argv);

  std::string identifier = icl_core::config::Getopt::instance().paramOpt("device-identifier");

  LOGGING_INFO(Gpu_voxels, "DistanceKinectDemo start. Device identifier: " << identifier << endl);

  // Generate a GPU-Voxels instance:
  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(map_dimensions.x, map_dimensions.y, map_dimensions.z, voxel_side_length);

  // Fire up the Kinect driver:
  Kinect* kinect = new Kinect(identifier);
  kinect->run();

  //Vis Helper
  gvl->addPrimitives(primitive_array::ePRIM_SPHERE, "measurementPoints");


  //PBA
  gvl->addMap(MT_DISTANCE_VOXELMAP, "pbaDistanceVoxmap");
  shared_ptr<DistanceVoxelMap> pbaDistanceVoxmap = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("pbaDistanceVoxmap"));

  //PBA map clone for visualization without artifacts
  gvl->addMap(MT_DISTANCE_VOXELMAP, "pbaDistanceVoxmapVisual");
  shared_ptr<DistanceVoxelMap> pbaDistanceVoxmapVisual = dynamic_pointer_cast<DistanceVoxelMap>(gvl->getMap("pbaDistanceVoxmapVisual"));

  pbaDistanceVoxmapVisual->clearMap();

  PointCloud myKinectCloud;
  myKinectCloud.add(kinect->getDataPtr());

  const Vector3f kinect_offsets(2, 0, 1);
  Matrix4f tf = Matrix4f::createFromRotationAndTranslation(Matrix3f::createFromRPY(-3.14/2.0, 0, 0), kinect_offsets);

  // Define two measurement points:
  std::vector<Vector3i> measurement_points;
  measurement_points.push_back(Vector3i(40, 100, 50));
  measurement_points.push_back(Vector3i(160, 100, 50));
  gvl->modifyPrimitives("measurementPoints", measurement_points, 5);

  LOGGING_INFO(Gpu_voxels, "start visualizing maps" << endl);
  while (true)
  {
    pbaDistanceVoxmap->clearMap();
    // Transform the Kinect cloud
    myKinectCloud.update(kinect->getDataPtr());
    myKinectCloud.transformSelf(&tf);

    // Insert the Kinect data (now in world coordinates) into the map
    pbaDistanceVoxmap->insertPointCloud(myKinectCloud, eBVM_OCCUPIED);
    // Calculate the distance map:
    pbaDistanceVoxmap->parallelBanding3D(m1, m2, m3, PBA_DEFAULT_M1_BLOCK_SIZE, PBA_DEFAULT_M2_BLOCK_SIZE, PBA_DEFAULT_M3_BLOCK_SIZE, 1);

    LOGGING_INFO(Gpu_voxels, "start cloning pbaDistanceVoxmap" << endl);
    pbaDistanceVoxmapVisual->clone(*(pbaDistanceVoxmap.get()));
    LOGGING_INFO(Gpu_voxels, "done cloning pbaDistanceVoxmap" << endl);

    gvl->visualizeMap("pbaDistanceVoxmapVisual");
    gvl->visualizePrimitivesArray("measurementPoints");

    // For the measurement points we query the clearance to the closest obstacle:
    thrust::device_ptr<DistanceVoxel> dvm_thrust_ptr(pbaDistanceVoxmap->getDeviceDataPtr());
    for(size_t i = 0; i < measurement_points.size(); i++)
    {
      int id = voxelmap::getVoxelIndexSigned(map_dimensions, measurement_points[i]);

      //get DistanceVoxel with closest obstacle information
      // DistanceVoxel dv = dvm_thrust_ptr[id]; // worked before Cuda9
      DistanceVoxel dv; //get DistanceVoxel with closest obstacle information
      cudaMemcpy(&dv, (dvm_thrust_ptr+id).get(), sizeof(DistanceVoxel), cudaMemcpyDeviceToHost);

      float metric_free_space = sqrtf(dv.squaredObstacleDistance(measurement_points[i])) * voxel_side_length;
      std::cout << "Obstacle @ " << dv.getObstacle() << " Voxel @ " << measurement_points[i] << " has a clearance of " << metric_free_space << "m." << std::endl;
    }
    usleep(100000);
  }

  LOGGING_INFO(Gpu_voxels, "shutting down" << endl);

  exit(EXIT_SUCCESS);
}

