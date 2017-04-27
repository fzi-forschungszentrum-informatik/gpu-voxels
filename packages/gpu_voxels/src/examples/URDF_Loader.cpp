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
 * \date    2015-05-27
 *
 * This little example program shows how to load and animate your ROS Robot.
 * Use "binvox" (external Software from Patrick Min)
 * to voxelize your meshes beforehand!
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/robot/urdf_robot/urdf_robot.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;
namespace bfs = boost::filesystem;

GpuVoxelsSharedPtr gvl;

void ctrlchandler(int)
{
  gvl.reset();
  exit(EXIT_SUCCESS);
}
void killhandler(int)
{
  gvl.reset();
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
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.01);// ==> 200 Voxels, each one is 10 mm in size so the map represents 2x2x2 meter

  // Add a map:
  //gvl->addMap(MT_BITVECTOR_OCTREE, "myOctree");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myBitVoxellist");

  // And a robot, generated from a ROS URDF file:
  gvl->addRobot("myUrdfRobot", "hollie/hollie.urdf", true);

  // Define some joint values:
  robot::JointValueMap min_joint_values;
  min_joint_values["hollie_plain_base_theta_joint"] = 0.0;
  min_joint_values["hollie_plain_base_x_joint"] = 3;
  min_joint_values["hollie_plain_base_y_joint"] = 2;
  min_joint_values["hollie_plain_lower_torso"] = -1.0;
  min_joint_values["hollie_plain_upper_torso"] = 0.0;
  min_joint_values["hollie_plain_neck_yaw"] = -0.4;
  min_joint_values["hollie_plain_left_arm_1_joint"] = 1.3;
  min_joint_values["hollie_plain_left_arm_2_joint"] = 0.0;
  min_joint_values["hollie_plain_left_arm_3_joint"] = 0.0;
  min_joint_values["hollie_plain_left_arm_4_joint"] = 0.0;
  min_joint_values["hollie_plain_right_arm_1_joint"] = -1.3;
  min_joint_values["hollie_plain_right_arm_2_joint"] = 0.0;
  min_joint_values["hollie_plain_right_arm_3_joint"] = 0.0;
  min_joint_values["hollie_plain_right_arm_4_joint"] = 0.0;

  robot::JointValueMap max_joint_values;
  max_joint_values["hollie_plain_base_theta_joint"] = 1.0;
  max_joint_values["hollie_plain_base_x_joint"] = 4;
  max_joint_values["hollie_plain_base_y_joint"] = 5;
  max_joint_values["hollie_plain_lower_torso"] = 0.5;
  max_joint_values["hollie_plain_upper_torso"] = 0.5;
  max_joint_values["hollie_plain_neck_yaw"] = 0.4;
  max_joint_values["hollie_plain_left_arm_1_joint"] = -1.5;
  max_joint_values["hollie_plain_left_arm_2_joint"] = 1.5;
  max_joint_values["hollie_plain_left_arm_3_joint"] = -1.5;
  max_joint_values["hollie_plain_left_arm_4_joint"] = -1.0;
  max_joint_values["hollie_plain_right_arm_1_joint"] = 0.7;
  max_joint_values["hollie_plain_right_arm_2_joint"] = -0.7;
  max_joint_values["hollie_plain_right_arm_3_joint"] = 2.0;
  max_joint_values["hollie_plain_right_arm_4_joint"] = 1.0;

  /*
   * Now we start the main loop, that will animate and the scene.
   */
  robot::JointValueMap myRobotJointValues;
  int counter = 0;
  int inc = +1;
  while(true)
  {
    myRobotJointValues = gpu_voxels::interpolateLinear(min_joint_values, max_joint_values,
                                                       0.01 * (counter += inc));
    if(counter > 100) inc = -1;
    if(counter < 1)   inc = +1;

    // update the robot joints:
    gvl->setRobotConfiguration("myUrdfRobot", myRobotJointValues);
    // insert the robot into the map:
    gvl->insertRobotIntoMap("myUrdfRobot", "myBitVoxellist", eBVM_OCCUPIED);

    std::cout << "Mem usage of Voxellist in Byte after insertion of robot: " << gvl->getMap("myBitVoxellist")->getMemoryUsage() << std::endl;

    // generate info:
    printf(".");
    fflush(stdout);

    // tell the visualier that the map has changed:
    gvl->visualizeMap("myBitVoxellist");

    usleep(30000);

//    // reset the map:
    gvl->clearMap("myBitVoxellist");
  }
}
