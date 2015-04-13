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
 *
 * This example program shows a simple collision check
 * between an animated Robot and a Static Map.
 * Be sure to press g in the viewer to draw the whole map.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <icl_core_config/Config.h>

#include <gpu_voxels/GpuVoxels.h>
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

  icl_core::logging::initialize(argc, argv);

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = new GpuVoxels(200, 200, 200, 0.01);

  /*
   * Now we add a map, that will represent the robot.
   * The robot is inserted with deterministic poses,
   * so a deterministic map is sufficient here.
   */
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myRobotMap");


  /*
   * A second map will represent the environment.
   * As it is captured by a sensor, this map is probabilistic.
   * We also have an a priori static map file in a PCD, so we
   * also load that into the map.
   * The PCD file is (in this example) retrieved via an environment variable
   * to access the directory where model files are stored in:
   * GPU_VOXELS_MODEL_PATH
   * The additional (optional) params shift the map to zero and then add
   * a offset to the loaded pointcloud.
   */
  gvl->addMap(MT_BITVECTOR_OCTREE, "myEnvironmentMap");

  if (!gvl->getMap("myEnvironmentMap")->insertPointcloudFromFile("pointcloud_0002.pcd", true,
                                                                 eVT_OCCUPIED, true, Vector3f(-6, -7.3, 0)))
  {
    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
  }

  /*
   * Of course, we need a robot. At this point, it would be helpful
   * to e.g. rely on KDL or something to parse DH params and be consistent
   * to your robot description.
   * In this example, we simply hardcode a robot:
   */

  // First, we load the robot geometry which contains 6 links:
  size_t rob_dim = 7;
  std::vector<std::string> paths_to_pointclouds(rob_dim);
  paths_to_pointclouds[0] = "arm_0_link.xyz";
  paths_to_pointclouds[1] = "arm_1_link.xyz";
  paths_to_pointclouds[2] = "arm_2_link.xyz";
  paths_to_pointclouds[3] = "arm_3_link.xyz";
  paths_to_pointclouds[4] = "arm_4_link.xyz";
  paths_to_pointclouds[5] = "arm_5_link.xyz";
  paths_to_pointclouds[6] = "arm_6_link.xyz";
  MetaPointCloud myRobotCloud(paths_to_pointclouds, true);

  std::vector<DHParameters> dh_params(rob_dim);
                                          // _d,_theta,_a,_alpha,_value
  dh_params[0] = DHParameters(0.0, 0.0, 0.0,     1.5708, 0.0); // build an arm from 6 segments
  dh_params[1] = DHParameters(0.0,  0.0, 0.35,  -3.1415, 0.0); //
  dh_params[2] = DHParameters(0.0,  0.0, 0.0,    1.5708, 0.0); //
  dh_params[3] = DHParameters(0.0,  0.0, 0.365, -1.5708, 0.0); //
  dh_params[4] = DHParameters(0.0,  0.0, 0.0,    1.5708, 0.0); //
  dh_params[5] = DHParameters(0.0,  0.0, 0.0,    0.0,    0.0); //
  dh_params[6] = DHParameters(0.0,  0.0, 0.0,    0.0,    0.0); //

  gvl->addRobot("myRobot", dh_params, myRobotCloud);

  // define a position for the robot
  gpu_voxels::Matrix4f new_base_pose;
  new_base_pose.a11 = 1;
  new_base_pose.a22 = 1;
  new_base_pose.a33 = 1;
  new_base_pose.a44 = 1;

  new_base_pose.a14 += 1.2;
  new_base_pose.a24 += 1.2;
  new_base_pose.a34 += 1.2;

  std::vector<float> myRobotJointValues(rob_dim, 0.0);

  // initialize the joint interpolation
  std::vector<float> min_joint_values(rob_dim, -1.0);
  std::vector<float> max_joint_values(rob_dim, 1.5);
  std::size_t counter = 0;
  const float ratio_delta = 0.01;

  /*
   * Now we enter "normal" operation
   * and make the robot move.
   */
  while(true)
  {
    /*
     * The robot moves and changes it's pose, so we "voxelize"
     * the links in every step and update the robot map.
     */
    LOGGING_INFO(Gpu_voxels, "Updating robot pose..." << endl);

    myRobotJointValues = gpu_voxels::CudaMath::interpolateLinear(min_joint_values, max_joint_values, ratio_delta * counter++);

    // we could also make it drive around:
    //new_base_pose.a14 += 0.02;
    //new_base_pose.a24 += 0.02;
    //new_base_pose.a34 += 0.00;

    gvl->updateRobotPose("myRobot", myRobotJointValues, &new_base_pose);

    gvl->insertRobotIntoMap("myRobot", "myRobotMap", eVT_OCCUPIED);

    /*
     * When the updates of the robot and the environment are
     * done, we can collide the maps and check for collisions.
     * The order of the maps is important here! The "smaller"
     * map should always be the first argument, as all occupied
     * Voxels from the first map will be looked up in the second map.
     * So, if you put a Voxelmap first, the GPU has to iterate over
     * the whole map in every step, to determine the occupied
     * Voxels. If you put an Octree first, the descend down to
     * the occupied Voxels is a lot more performant.
     */
    LOGGING_INFO(
        Gpu_voxels,
        "Collsions: " << gvl->getMap("myEnvironmentMap")->collideWith(gvl->getMap("myRobotMap")) << endl);

    // visualize both maps
    gvl->visualizeMap("myRobotMap");
    gvl->visualizeMap("myEnvironmentMap");

    usleep(100000);

    // We assume that the robot will be updated in the next loop, so we clear the map.
    gvl->clearMap("myRobotMap");
  }

}
