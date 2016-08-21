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
  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 100, 0.02);

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
   * we declare it as a robot and model a pan-tilt-unit.
   */
  Kinect* kinect = new Kinect(identifier);
  kinect->run();
  std::vector<std::string> kinect_link_names(6);
  kinect_link_names[0] = "z_translation";
  kinect_link_names[1] = "y_translation";
  kinect_link_names[2] = "x_translation";
  kinect_link_names[3] = "pan";
  kinect_link_names[4] = "tilt";
  kinect_link_names[5] = "kinect";

  std::vector<robot::DHParameters> kinect_dh_params(6);
  kinect_dh_params[0] = robot::DHParameters(0.0,  0.0,    0.0,   -1.5708, 0.0, robot::PRISMATIC); // Params for Y translation
  kinect_dh_params[1] = robot::DHParameters(0.0, -1.5708, 0.0,   -1.5708, 0.0, robot::PRISMATIC); // Params for X translation
  kinect_dh_params[2] = robot::DHParameters(0.0,  1.5708, 0.0,    1.5708, 0.0, robot::PRISMATIC); // Params for Pan axis
  kinect_dh_params[3] = robot::DHParameters(0.0,  1.5708, 0.0,    1.5708, 0.0, robot::REVOLUTE);  // Params for Tilt axis
  kinect_dh_params[4] = robot::DHParameters(0.0,  0.0,    0.0,   -3.1415, 0.0, robot::REVOLUTE);  // Params for Kinect
  kinect_dh_params[5] = robot::DHParameters(0.0,  0.0,    0.0,    0.0,    0.0, robot::REVOLUTE);  // Pseudo Param

  robot::JointValueMap kinect_joints;
  kinect_joints["z_translation"] = 0.6; // moves along the Z axis
  kinect_joints["y_translation"] = 1.0; // moves along the Y Axis
  kinect_joints["x_translation"] = 1.0; // moves along the X Axis
  kinect_joints["pan"]  = -0.7;
  kinect_joints["tilt"] = 0.5;

  std::vector<Vector3f> kinect_pc(640*480);
  MetaPointCloud myKinectCloud;
  myKinectCloud.addCloud(kinect_pc, true, kinect_link_names[5]);

  gvl->addRobot("kinectData", kinect_link_names, kinect_dh_params, myKinectCloud);


  /*
   * Of course, we need a robot. At this point, you can choose between
   * describing your robot via ROS URDF or via conventional DH parameter.
   * In this example, we simply hardcode a DH robot:
   */

  // First, we load the robot geometry which contains 9 links with 7 geometries:
  // Geometries are required to have the same names as links, if they should get transformed.
  std::vector<std::string> linknames(10);
  std::vector<std::string> paths_to_pointclouds(7);
  linknames[0] = "z_translation";
  linknames[1] = "y_translation";
  linknames[2] = "x_translation";
  linknames[3] = paths_to_pointclouds[0] = "hollie/arm_0_link.xyz";
  linknames[4] = paths_to_pointclouds[1] = "hollie/arm_1_link.xyz";
  linknames[5] = paths_to_pointclouds[2] = "hollie/arm_2_link.xyz";
  linknames[6] = paths_to_pointclouds[3] = "hollie/arm_3_link.xyz";
  linknames[7] = paths_to_pointclouds[4] = "hollie/arm_4_link.xyz";
  linknames[8] = paths_to_pointclouds[5] = "hollie/arm_5_link.xyz";
  linknames[9] = paths_to_pointclouds[6] = "hollie/arm_6_link.xyz";

  std::vector<robot::DHParameters> dh_params(10);
                                   // _d,  _theta,  _a,   _alpha, _value, _type
  dh_params[0] = robot::DHParameters(0.0,  0.0,    0.0,   -1.5708, 0.0, robot::PRISMATIC); // Params for Y translation
  dh_params[1] = robot::DHParameters(0.0, -1.5708, 0.0,   -1.5708, 0.0, robot::PRISMATIC); // Params for X translation
  dh_params[2] = robot::DHParameters(0.0,  1.5708, 0.0,    1.5708, 0.0, robot::PRISMATIC); // Params for first Robot axis (visualized by 0_link)
  dh_params[3] = robot::DHParameters(0.0,  1.5708, 0.0,    1.5708, 0.0, robot::REVOLUTE);  // Params for second Robot axis (visualized by 1_link)
  dh_params[4] = robot::DHParameters(0.0,  0.0,    0.35,  -3.1415, 0.0, robot::REVOLUTE);  //
  dh_params[5] = robot::DHParameters(0.0,  0.0,    0.0,    1.5708, 0.0, robot::REVOLUTE);  //
  dh_params[6] = robot::DHParameters(0.0,  0.0,    0.365, -1.5708, 0.0, robot::REVOLUTE);  //
  dh_params[7] = robot::DHParameters(0.0,  0.0,    0.0,    1.5708, 0.0, robot::REVOLUTE);  //
  dh_params[8] = robot::DHParameters(0.0,  0.0,    0.0,    0.0,    0.0, robot::REVOLUTE);  // Params for last Robot axis (visualized by 6_link)
  dh_params[9] = robot::DHParameters(0.0,  0.0,    0.0,    0.0,    0.0, robot::REVOLUTE);  // Params for the not viusalized tool

  gvl->addRobot("myRobot", linknames, dh_params, paths_to_pointclouds, true);

  // initialize the joint interpolation
  std::size_t counter = 0;
  const float ratio_delta = 0.02;

  robot::JointValueMap min_joint_values;
  min_joint_values["z_translation"] = 0.0; // moves along the Z axis
  min_joint_values["y_translation"] = 0.5; // moves along the Y Axis
  min_joint_values["x_translation"] = 0.5; // moves along the X Axis
  min_joint_values["hollie/arm_0_link.xyz"] = 1.0;
  min_joint_values["hollie/arm_1_link.xyz"] = 1.0;
  min_joint_values["hollie/arm_2_link.xyz"] = 1.0;
  min_joint_values["hollie/arm_3_link.xyz"] = 1.0;
  min_joint_values["hollie/arm_4_link.xyz"] = 1.0;
  min_joint_values["hollie/arm_5_link.xyz"] = 1.0;
  min_joint_values["hollie/arm_6_link.xyz"] = 1.0;

  robot::JointValueMap max_joint_values;
  max_joint_values["z_translation"] = 0.0; // moves along the Z axis
  max_joint_values["y_translation"] = 2.5; // moves along the Y axis
  max_joint_values["x_translation"] = 2.5; // moves along the X Axis
  max_joint_values["hollie/arm_0_link.xyz"] = 1.5;
  max_joint_values["hollie/arm_1_link.xyz"] = 1.5;
  max_joint_values["hollie/arm_2_link.xyz"] = 1.5;
  max_joint_values["hollie/arm_3_link.xyz"] = 1.5;
  max_joint_values["hollie/arm_4_link.xyz"] = 1.5;
  max_joint_values["hollie/arm_5_link.xyz"] = 1.5;
  max_joint_values["hollie/arm_6_link.xyz"] = 1.5;

  const int num_swept_volumes = 50;// < BIT_VECTOR_LENGTH;

  /*
   * SWEPT VOLUME:
   * The robot moves and changes it's pose, so we "voxelize"
   * the links in every step and insert it into the robot map.
   * As the map is not cleared, this will generate a sweep.
   * The ID within the sweep is incremented with the single poses
   * so we can later identify, which pose created a collision.
   */
  LOGGING_INFO(Gpu_voxels, "Generating Swept Volume..." << endl);
  robot::JointValueMap myRobotJointValues;
  for (int i = 0; i < num_swept_volumes; ++i)
  {

    myRobotJointValues = gpu_voxels::interpolateLinear(min_joint_values, max_joint_values,
                                                       ratio_delta * counter++);

    gvl->setRobotConfiguration("myRobot", myRobotJointValues);
    BitVoxelMeaning v = BitVoxelMeaning(eBVM_SWEPT_VOLUME_START + 1 + i);
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
    // Insert Kinect data (in cam-coordinate system)
    gvl->updateRobotPart("kinectData", "kinect", kinect->getDataPtr());
    // Call setRobotConfiguration to trigger transformation of Kinect data:
    gvl->setRobotConfiguration("kinectData", kinect_joints);
    // Insert the Kinect data (now in world coordinates) into the map
    gvl->insertRobotIntoMap("kinectData", "myEnvironmentMap", eBVM_OCCUPIED);

    size_t num_cols = 0;
    BitVectorVoxel collision_types;
    num_cols = gvl->getMap("myEnvironmentMap")->as<NTree::GvlNTreeDet>()->collideWithTypes(gvl->getMap("myRobotMap")->as<voxelmap::BitVectorVoxelMap>(), collision_types, 1.0f);
    LOGGING_INFO(Gpu_voxels, "Collsions: " << num_cols << endl);

    printf("Voxel types in collision:\n");
    DrawTypes draw_types;
    for(size_t i = 0; i < BIT_VECTOR_LENGTH; ++i)
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
