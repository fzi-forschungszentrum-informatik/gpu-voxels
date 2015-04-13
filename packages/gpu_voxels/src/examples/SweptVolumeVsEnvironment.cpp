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
 * \author  Andreas Hermann
 * \date    2014-06-08
 *
 * This program demonstrates the creation of a Swept Volume.
 * To view it appropriately in the viewer, please make sure,
 * that external visibility triggering of SweptVolumes is activated (Press o).
 * Also check, that the whole maps are drwan (Press g).
 *
 * Then you should be able to provoque a collision between the swept volume
 * and the Kinect Pointcloud.
 * The visualizer will then render only the subvolume of the Sweep that
 * lies in collision.
 *
 * In a real application you can use this information to determine the
 * remaing time to impact or trigger a very specific replanning of the
 * motion.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/Kinect.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;

GpuVoxels* gvl;

void ctrlchandler(int)
{
  delete gvl;
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  delete gvl;
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

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = new GpuVoxels(500, 500, 200, 0.01);

  /*
   * Now we add a map, that will represent the robot.
   * The robot is inserted with deterministic poses,
   * so a deterministic map is sufficient here.
   */
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myRobotMap");

  /*
   * A second map will represent the environment.
   * As it is captured by a sensor, this map is probabilistic.
   */
  gvl->addMap(MT_BITVECTOR_OCTREE, "myEnvironmentMap");

  /*
   * Lets create a kinect driver and an according pointcloud.
   * To allow easy transformation of the Kinect pose,
   * we declare it as a robot.
   */
  Kinect* kinect = new Kinect(identifier);
  kinect->run();
  std::vector<DHParameters> kinect_dh_params(1);
  kinect_dh_params[0] = DHParameters(0.0, 0.0, 0.0, 0.0, 0.0);
  std::vector<uint32_t> sizes(1, 640*480);
  MetaPointCloud myKinectCloud(sizes);
  gvl->addRobot("kinectData", kinect_dh_params, myKinectCloud);

  /*
   * Of course, we need another robot. At this point, it would be helpful
   * to e.g. rely on KDL or something to parse DH params and be consistent
   * to your robot description.
   * In this example, we simply hardcode a robot.
   * The environment variable GPU_VOXELS_MODEL_PATH is checked
   * to ensure that the robot parts can be loaded.
   */

  size_t rob_dim = 7;
  // this loads the segment models from pointclouds
  std::vector<std::string> paths_to_pointclouds(rob_dim);
  paths_to_pointclouds[0] = "arm_0_link.xyz";
  paths_to_pointclouds[1] = "arm_1_link.xyz";
  paths_to_pointclouds[2] = "arm_2_link.xyz";
  paths_to_pointclouds[3] = "arm_3_link.xyz";
  paths_to_pointclouds[4] = "arm_4_link.xyz";
  paths_to_pointclouds[5] = "arm_5_link.xyz";
  paths_to_pointclouds[6] = "arm_6_link.xyz";
  MetaPointCloud myRobotCloud(paths_to_pointclouds, true);

  // this is the DH description of the robots kinematic
  std::vector<DHParameters> dh_params(rob_dim);
                                          // _d,_theta,_a,_alpha,_value
  dh_params[0] = DHParameters(0.0, 0.0, 0.0,     1.5708, 0.0); // build an arm from 6 segments
  dh_params[1] = DHParameters(0.0,  0.0, 0.35,  -3.1415, 0.0);
  dh_params[2] = DHParameters(0.0,  0.0, 0.0,    1.5708, 0.0);
  dh_params[3] = DHParameters(0.0,  0.0, 0.365, -1.5708, 0.0);
  dh_params[4] = DHParameters(0.0,  0.0, 0.0,    1.5708, 0.0);
  dh_params[5] = DHParameters(0.0,  0.0, 0.0,    0.0,    0.0);
  dh_params[6] = DHParameters(0.0,  0.0, 0.0,    0.0,    0.0);

  // now we add the robot to the management
  gvl->addRobot("myRobot", dh_params, myRobotCloud);

  // to allow a motion of the robot we need some poses and joint angles:
  gpu_voxels::Matrix4f new_base_pose;
  new_base_pose.a11 = 1;
  new_base_pose.a22 = 1;
  new_base_pose.a33 = 1;
  new_base_pose.a44 = 1;
  new_base_pose.a14 += 0;
  new_base_pose.a24 += 2;
  new_base_pose.a34 += 0.5;

  std::vector<float> myRobotJointValues(rob_dim, 0.0);

  // initialize the joint interpolation
  std::vector<float> min_joint_values(rob_dim, -1.0);
  std::vector<float> max_joint_values(rob_dim, 1.5);
  std::size_t counter = 0;
  const float ratio_delta = 0.02;
  const int num_swept_volumes = 50;//voxelmap::BIT_VECTOR_LENGTH;

  /*
   * SWEPT VOLUME:
   * The robot moves and changes it's pose, so we "voxelize"
   * the links in every step and insert it into the robot map.
   * As the map is not cleared, this will generate a sweep.
   * The ID within the sweep is incremented with the single poses
   * so we can later identify, which pose created a collision.
   */
  LOGGING_INFO(Gpu_voxels, "Generating Swept Volume..." << endl);
  for (int i = 0; i < num_swept_volumes; ++i)
  {
    myRobotJointValues = gpu_voxels::CudaMath::interpolateLinear(min_joint_values, max_joint_values,
                                                                 ratio_delta * counter++);
    new_base_pose.a14 += 0.05;
    new_base_pose.a24 += 0.05;
    new_base_pose.a34 += 0.00;

    gvl->updateRobotPose("myRobot", myRobotJointValues, &new_base_pose);
    VoxelType v = VoxelType(eVT_SWEPT_VOLUME_START + 1 + i);
    gvl->insertRobotIntoMap("myRobot", "myRobotMap", v);
  }


  /*
   * MAIN LOOP:
   * In this loop we update the Kinect Pointcloud
   * and collide it with the Swept-Volume of the robot.
   */
  LOGGING_INFO(Gpu_voxels, "Starting collision detection..." << endl);
  while (true)
  {
    std::vector<float> joints(1, 0.0);
    gpu_voxels::Matrix4f kinect_base_pose;

    // Rotate the kinect 90 Degrees about the X-Axis:
    kinect_base_pose = gpu_voxels::roll(M_PI_2);
    kinect_base_pose.a14 = 1;
    kinect_base_pose.a24 = 1;
    kinect_base_pose.a34 = 1;
    kinect_base_pose.a44 = 1;

    gvl->updateRobotPose("kinectData", joints, &kinect_base_pose);
    gvl->updateRobotPart("kinectData", 0, kinect->getDataPtr());
    gvl->insertRobotIntoMap("kinectData", "myEnvironmentMap", eVT_OCCUPIED);

    size_t num_cols = 0;
    voxelmap::BitVectorVoxel collision_types;
    num_cols = gvl->getMap("myEnvironmentMap")->collideWithTypes(gvl->getMap("myRobotMap"), collision_types, 1.0f);
    LOGGING_INFO(Gpu_voxels, "Collsions: " << num_cols << endl);

    printf("Voxel types in collision:\n");
    DrawTypes draw_types;
    for(size_t i = 0; i < voxelmap::BIT_VECTOR_LENGTH; ++i)
    {
      if(collision_types.bitVector().getBit(i))
      {
        draw_types.draw_types[i] = 1;
        printf("%lu; ", i);
      }
    }
    printf("\n");

    // this informs the visualizer which Sub-Volumes should be rendered
    gvl->getVisualization("myRobotMap")->setDrawTypes(draw_types);


    // tell the visualizer that the data has changed.
    gvl->visualizeMap("myRobotMap");
    gvl->visualizeMap("myEnvironmentMap");

    usleep(10000);

    // We only clear the environment to update it with new Kinect data.
    // The robot maps stays static to not loose the Sweeps.
    gvl->clearMap("myEnvironmentMap");
  }

}
