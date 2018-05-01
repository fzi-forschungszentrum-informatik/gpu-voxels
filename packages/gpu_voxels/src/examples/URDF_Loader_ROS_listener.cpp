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

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/Pose.h>

using namespace gpu_voxels;
namespace bfs = boost::filesystem;

GpuVoxelsSharedPtr gvl;

void ctrlchandler(int)
{
  ros::shutdown();
}
void killhandler(int)
{
  ros::shutdown();
}

robot::JointValueMap myRobotJointValues;
Vector3f object_position(0.1, 0.15, 0.15);

void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  //std::cout << "Got JointStateMessage" << std::endl;
  gvl->clearMap("myHandVoxellist");

  for(size_t i = 0; i < msg->name.size(); i++)
  {
    myRobotJointValues[msg->name[i]] = msg->position[i];
  }
  // update the robot joints:
  gvl->setRobotConfiguration("myUrdfRobot", myRobotJointValues);
  // insert the robot into the map:
  gvl->insertRobotIntoMap("myUrdfRobot", "myHandVoxellist", eBVM_OCCUPIED);
}

void obstaclePoseCallback(const geometry_msgs::Pose::ConstPtr& msg)
{
  std::cout << "Got PoseMessage" << std::endl;
  gvl->clearMap("myObjectVoxelmap");

  object_position.x = msg->position.x;
  object_position.y = msg->position.y;
  object_position.z = msg->position.z;

  gvl->insertPointCloudFromFile("myObjectVoxelmap", "hollie/hals_vereinfacht.binvox", true,
                                eBVM_OCCUPIED, false, object_position, 0.3);
}

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  ros::init(argc, argv, "gpu_voxels");

  icl_core::logging::initialize(argc, argv);

  /*
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.001); // ==> 200 Voxels, each one is 1 mm in size so the map represents 20x20x20 centimeter

  // Add a map:
  gvl->addMap(MT_PROBAB_VOXELMAP, "myObjectVoxelmap");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myHandVoxellist");

  // And a robot, generated from a ROS URDF file:
  gvl->addRobot("myUrdfRobot", "schunk_svh/svh-standalone.urdf", true);

  ros::NodeHandle n;
  ros::Subscriber sub1 = n.subscribe("joint_states", 1, jointStateCallback);
  ros::Subscriber sub2 = n.subscribe("obstacle_pose", 1, obstaclePoseCallback);

  // A an obstacle that can be moved with the ROS callback
  gvl->insertPointCloudFromFile("myObjectVoxelmap", "hollie/hals_vereinfacht.binvox", true,
                                eBVM_OCCUPIED, false, object_position, 0.3);
  // update the robot joints:
  gvl->setRobotConfiguration("myUrdfRobot", myRobotJointValues);
  // insert the robot into the map:
  gvl->insertRobotIntoMap("myUrdfRobot", "myHandVoxellist", eBVM_OCCUPIED);
  /*
   * Now we start the main loop, that will read ROS messages and update the Robot.
   */
  BitVectorVoxel bits_in_collision;
  size_t num_colls;
  while(ros::ok())
  {
    ros::spinOnce();

    num_colls = gvl->getMap("myHandVoxellist")->as<voxellist::BitVectorVoxelList>()->collideWithTypes(gvl->getMap("myObjectVoxelmap")->as<voxelmap::ProbVoxelMap>(), bits_in_collision);

    std::cout << "Detected " << num_colls << " collisions " << std::endl;
    //std::cout << "with bits \n" << bits_in_collision << std::endl;

    // tell the visualier that the map has changed:
    gvl->visualizeMap("myHandVoxellist");
    gvl->visualizeMap("myObjectVoxelmap");

    usleep(30000);
  }

  gvl.reset();
  exit(EXIT_SUCCESS);
}
