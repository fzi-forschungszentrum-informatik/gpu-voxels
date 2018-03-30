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
  gvl->initialize(200, 200, 200, 0.02);// ==> 200 Voxels, each one is 20 mm in size so the map represents 4x4x4 meter

  // Add a map:
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myBitVoxelMap");

  // And a robot, generated from a ROS URDF file:
  gvl->addRobot("myUrdfRobot", "ur10_coarse/ur10_joint_limited_robot.urdf", true);

  // Define some joint values:
  robot::JointValueMap min_joint_values;
  min_joint_values["shoulder_pan_joint"]  = 0.0;
  min_joint_values["shoulder_lift_joint"] = -1.0;
  min_joint_values["elbow_joint"]         = -2.0;
  min_joint_values["wrist_1_joint"]       = 0.0;
  min_joint_values["wrist_2_joint"]       = 0.0;
  min_joint_values["wrist_3_joint"]       = 0.0;

  robot::JointValueMap max_joint_values;
  max_joint_values["shoulder_pan_joint"]  = 0.1;
  max_joint_values["shoulder_lift_joint"] = -1.1;
  max_joint_values["elbow_joint"]         = -4.1;
  max_joint_values["wrist_1_joint"]       = 0.1;
  max_joint_values["wrist_2_joint"]       = 0.1;
  max_joint_values["wrist_3_joint"]       = 0.1;
  /*
   * Now we start the main loop, that will animate and the scene.
   */
  robot::JointValueMap myRobotJointValues;
  int counter = 0;
  int inc = +1;


  // Create Bitmasks and meanings:
  size_t num_links(gvl->getRobot("myUrdfRobot")->getTransformedClouds()->getNumberOfPointclouds());

  std::cout << "num_links = " << num_links << std::endl;

  std::vector<BitVoxelMeaning> voxel_meanings(num_links);

  size_t arbitrary_offset = 5; // arbitrary offset for demonstration purposes. Usefull if you need more than one robot.

  for(size_t link = 0; link < num_links; ++link)
  {
    voxel_meanings[link] = BitVoxelMeaning(arbitrary_offset + link);
  }

  std::vector<BitVector<BIT_VECTOR_LENGTH> > collision_masks(num_links);
  BitVector<BIT_VECTOR_LENGTH> no_coll_mask;
  BitVector<BIT_VECTOR_LENGTH> all_coll_mask;
  all_coll_mask = ~no_coll_mask; // set all bits to true
  all_coll_mask.clearBit(eBVM_COLLISION);

  //  0 = Base
  collision_masks[0] = all_coll_mask;
  collision_masks[0].clearBit(arbitrary_offset + 0); // disable collisions of segment with itself
  collision_masks[0].clearBit(arbitrary_offset + 3); // disable collisions with shoulder

  //  1 = Forearm
  collision_masks[1] = all_coll_mask;
  collision_masks[1].clearBit(arbitrary_offset + 1); // disable collisions of segment with itself
  collision_masks[1].clearBit(arbitrary_offset + 3); // disable collisions with upper arm
  collision_masks[1].clearBit(arbitrary_offset + 4); // disable collisions with wrist
  collision_masks[1].clearBit(arbitrary_offset + 5); // disable collisions with wrist
  collision_masks[1].clearBit(arbitrary_offset + 6); // disable collisions with wrist

  //  2 = Shoulder
  collision_masks[2] = all_coll_mask;
  collision_masks[2].clearBit(arbitrary_offset + 2); // disable collisions of segment with itself
  collision_masks[2].clearBit(arbitrary_offset + 0); // disable collisions with base
  collision_masks[2].clearBit(arbitrary_offset + 3); // disable collisions with upper arm

  //  3 = Upper Arm
  collision_masks[3] = all_coll_mask;
  collision_masks[3].clearBit(arbitrary_offset + 3); // disable collisions of segment with itself
  collision_masks[3].clearBit(arbitrary_offset + 2); // disable collisions with shoulder
  collision_masks[3].clearBit(arbitrary_offset + 1); // disable collisions with forearm

  // 4 = Wrist
  collision_masks[4] = all_coll_mask;
  collision_masks[4].clearBit(arbitrary_offset + 4); // disable collisions of segment with itself
  collision_masks[4].clearBit(arbitrary_offset + 5); // disable collisions wrist
  collision_masks[4].clearBit(arbitrary_offset + 1); // disable collisions with forearm
  collision_masks[4].clearBit(arbitrary_offset + 6); // disable collisions wrist


  // 5 = Wrist
  collision_masks[5] = all_coll_mask;
  collision_masks[5].clearBit(arbitrary_offset + 5); // disable collisions of segment with itself
  collision_masks[5].clearBit(arbitrary_offset + 6); // disable collisions with wrist
  collision_masks[5].clearBit(arbitrary_offset + 4); // disable collisions with wrist
  collision_masks[5].clearBit(arbitrary_offset + 1); // disable collisions with forearm

  // 6 = Wrist
  collision_masks[6] = all_coll_mask;
  collision_masks[6].clearBit(arbitrary_offset + 6); // disable collisions of segment with itself
  collision_masks[6].clearBit(arbitrary_offset + 4); // disable collisions with wrist
  collision_masks[6].clearBit(arbitrary_offset + 5); // disable collisions with wrist
  collision_masks[6].clearBit(arbitrary_offset + 1); // disable collisions with forearm


  BitVector<BIT_VECTOR_LENGTH> colliding_meanings;

  while(true)
  {
    myRobotJointValues = gpu_voxels::interpolateLinear(min_joint_values, max_joint_values,
                                                       0.01 * (counter += inc));
    if(counter > 100) inc = -1;
    if(counter < 1)   inc = +1;

    // update the robot joints:
    gvl->setRobotConfiguration("myUrdfRobot", myRobotJointValues);

    // Insert the robot into the map and check for self collisions:

    // We could call the insertion function without additional param, but then some GPU structures are reallocated in every call:
    //if(gvl->insertRobotIntoMapSelfCollAware("myUrdfRobot", "myBitVoxellist"))
    //{
    //  std::cout << "Self collision occurred!" << std::endl;
    //}

    // Therefore we hand in external structures, which suppresses the reallocation:
    if(gvl->insertRobotIntoMapSelfCollAware("myUrdfRobot", "myBitVoxelMap", voxel_meanings, collision_masks, &colliding_meanings))
    {
      std::cout << "Self collision occurred in meanings " << colliding_meanings << std::endl;
    }else{
      std::cout << "No Self collision occurred" << std::endl;
    }

//    gvl->insertMetaPointCloudIntoMap(gvl->getRobot("myUrdfRobot")->getTransformedClouds(), "myBitVoxelMap", voxel_meanings);

    // tell the visualier that the map has changed:
    gvl->visualizeMap("myBitVoxelMap");

    usleep(100000);

    // reset the map:
    gvl->clearMap("myBitVoxelMap");
  }
}
