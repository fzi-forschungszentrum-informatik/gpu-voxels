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
 * \date    2016-08-07
 *
 * This little example program shows how to lookup
 * and publish ROS TFs
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/PointCloud.h>
#include <gpu_voxels/helpers/PointcloudFileHandler.h>
#include <gpu_voxels/helpers/tfHelper.h>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>


using namespace gpu_voxels;

GpuVoxelsSharedPtr gvl;

void ctrlchandler(int)
{
  ros::shutdown();
}
void killhandler(int)
{
  ros::shutdown();
}


voxelmap::BitVectorVoxelMap* my_map;

int main(int argc, char* argv[])
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  ros::init(argc, argv, "gpu_voxels");
  gpu_voxels::tfHelper my_tf_helper;

  icl_core::logging::initialize(argc, argv);

  gvl = GpuVoxels::getInstance();
  gvl->initialize(200, 200, 200, 0.01); // ==> 200 Voxels, each one is 1 mm in size so the map represents 20x20x20 centimeter

  // Add a map:
  gvl->addMap(MT_BITVECTOR_VOXELMAP, "myObjectVoxelmap");

  my_map = gvl->getMap("myObjectVoxelmap")->as<voxelmap::BitVectorVoxelMap>();

  PointCloud coord_system_object("coordinate_system_100.binvox");
  PointCloud coord_system_object_transformed;

  Matrix4f trafo;
  Matrix3f rot_from_rpy1;
  Matrix3f rot_from_rpy2;

  Vector3f rpy1;
  Vector3f rpy2;

  while(ros::ok())
  {
    ros::spinOnce();

    my_tf_helper.lookup("world", "demo_tf_1", trafo);

    coord_system_object.transform(&trafo, &coord_system_object_transformed);

    rpy1 = trafo.getRotation().toRPY(1);
    rot_from_rpy1 = Matrix3f::createFromRPY(rpy1);
    rpy2 = trafo.getRotation().toRPY(2);
    rot_from_rpy2 = Matrix3f::createFromRPY(rpy2);

    my_map->clearMap();
    my_map->insertPointCloud(coord_system_object_transformed, eBVM_OCCUPIED);
    gvl->visualizeMap("myObjectVoxelmap");

    // This is just for demonstration purposes:
    my_tf_helper.publish(Matrix4f::createFromRotationAndTranslation(rot_from_rpy1, trafo.getTranslation()), "world", "demo_tf_rpy_1");
    my_tf_helper.publish(Matrix4f::createFromRotationAndTranslation(rot_from_rpy2, trafo.getTranslation()), "world", "demo_tf_rpy_2");

    usleep(30000);
  }

  gvl.reset();
  exit(EXIT_SUCCESS);
}
