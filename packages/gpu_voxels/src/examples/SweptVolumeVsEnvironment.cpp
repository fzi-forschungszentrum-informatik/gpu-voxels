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
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/Kinect.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/file_handling.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>
#include <gpu_voxels/helpers/binvox_handling.h>

using namespace gpu_voxels;
namespace bfs = boost::filesystem;

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

  Kinect* kinect = new Kinect();

  /*!
   * First, we generate an API class, which defines the
   * volume of our space and the resolution.
   * Be careful here! The size is limited by the memory
   * of your GPU. Even if an empty Octree is small, a
   * Voxelmap will always require the full memory.
   */
  gvl = new GpuVoxels(500, 500, 200, 0.01);

  /*!
   * Now we add a map, that will represent the robot.
   * The robot is inserted with deterministic poses,
   * so a deterministic map is sufficient here.
   */
  gvl->addMap(MT_BIT_VOXELMAP, "myRobotMap");

  /*!
   * As we also want to plan the mobile platform,
   * we add another map, consisting only of a list
   * of voxels, that represent the robot
   */
  //gvl->addMap(eGVL_VOXELMAP, "myPlanningMap");

  /*!
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
  gvl->addMap(MT_OCTREE, "myEnvironmentMap");

  bfs::path model_path;
  if (!file_handling::getGpuVoxelsPath(model_path))
  {
    LOGGING_ERROR(Gpu_voxels, "The environment variable 'GPU_VOXELS_MODEL_PATH' could not be read. Did you set it?" << endl);
    return -1; // exit here.
  }

//  bfs::path pcd_file = bfs::path(model_path / "pointcloud_0002.pcd");
//  if (!gvl->getMap("myEnvironmentMap")->insertPCD(pcd_file.generic_string(), eVT_OCCUPIED,
//                                                  true, Vector3f(-5, -5, 0)))
//  {
//    LOGGING_WARNING(Gpu_voxels, "Could not insert the PCD file..." << endl);
//  }

  /*!
   * To get environment data, we register a callback from
   * a RGBD camera to insert pointcloud data into the map.
   */
  //kinect::registerCallback(gvl->InsertNewCamData("myEnvironmentMap"));
  /*!
   * Of course, we need a robot. At this point, it would be helpful
   * to e.g. rely on KDL or something to parse DH params and be consistent
   * to your robot description.
   * In this example, we simply hardcode a robot:
   */

  // First, we load the robot geometry which contains 7 links:
  LOGGING_INFO(Gpu_voxels, "Search path of the models is: " << model_path.generic_string() << endl);


  size_t rob_dim = 7;
  std::vector<std::string> paths_to_pointclouds(rob_dim);
  paths_to_pointclouds[0] = bfs::path(model_path / "hollie_plain_left_arm_0_link.xyz").generic_string();
  paths_to_pointclouds[1] = bfs::path(model_path / "hollie_plain_left_arm_1_link.xyz").generic_string();
  paths_to_pointclouds[2] = bfs::path(model_path / "hollie_plain_left_arm_2_link.xyz").generic_string();
  paths_to_pointclouds[3] = bfs::path(model_path / "hollie_plain_left_arm_3_link.xyz").generic_string();
  paths_to_pointclouds[4] = bfs::path(model_path / "hollie_plain_left_arm_4_link.xyz").generic_string();
  paths_to_pointclouds[5] = bfs::path(model_path / "hollie_plain_left_arm_5_link.xyz").generic_string();
  paths_to_pointclouds[6] = bfs::path(model_path / "hollie_plain_left_arm_6_link.xyz").generic_string();
  MetaPointCloud myRobotCloud(paths_to_pointclouds);

  // then we add another link, that contains data from an onboard sensor:
  //rob_dim++;
  //std::vector<Vector3f> kinect_frame(640*480, 0.0);
  //myRobotCloud.addCloud(kinect_frame); // this allocates mem for a kinect frame

  std::vector<KinematicLink::DHParameters> dh_params(rob_dim);
                                          // _d,_theta,_a,_alpha,_value
  dh_params[0] = KinematicLink::DHParameters(0.0, 0.0, 0.0,     1.5708, 0.0); // build an arm from 6 segments
  dh_params[1] = KinematicLink::DHParameters(0.0,  0.0, 0.35,  -3.1415, 0.0); //
  dh_params[2] = KinematicLink::DHParameters(0.0,  0.0, 0.0,    1.5708, 0.0); //
  dh_params[3] = KinematicLink::DHParameters(0.0,  0.0, 0.365, -1.5708, 0.0); //
  dh_params[4] = KinematicLink::DHParameters(0.0,  0.0, 0.0,    1.5708, 0.0); //
  dh_params[5] = KinematicLink::DHParameters(0.0,  0.0, 0.0,    0.0,    0.0); //
  dh_params[6] = KinematicLink::DHParameters(0.0,  0.0, 0.0,    0.0,    0.0); // this represents the angle to the kinect pointcloud CS
  // the last dh param was initialized with 0

  gvl->addRobot("myRobot", dh_params, myRobotCloud);



  std::vector<KinematicLink::DHParameters> kinect_dh_params(1);
  kinect_dh_params[0] = KinematicLink::DHParameters(0.0, 0.0, 0.0, 0.0, 0.0);
  std::vector<uint32_t> sizes(1, 640*480);
  MetaPointCloud myKinectCloud(sizes);
  gvl->addRobot("kinectData", kinect_dh_params, myKinectCloud);

  // ============ BEGIN TESTING =============
  //gvl->addMap(MT_PROBAB_VOXELMAP, "mySecondRobotMap");

//  if(!gvl->getMap("mySecondRobotMap")->insertPCD("/home/hermann/pcdrobotiklabor/pc_file5.pcd", eVT_OCCUPIED, true))
//  {
//    std::cout << "Could not insert the PCD file..." << std::endl;
//  }

//  if(!gvl->getMap("myRobotMap")->insertPCD("/home/hermann/pcdrobotiklabor/pc_file5.pcd", eVT_OCCUPIED, true, Vector3f(1,1,0)))
//  {
//    std::cout << "Could not insert the PCD file..." << std::endl;
//  }

//  std::cout << "Collsions: " << gvl->getMap("myRobotMap")->collideWith(gvl->getMap("mySecondRobotMap")) << std::endl;

//  paths_to_pointclouds.clear();
//  paths_to_pointclouds.resize(1);
//  paths_to_pointclouds[0] = bfs::path(model_path / "helmet1_3.binvox").generic_string();
//  MetaPointCloud myBinvoxCloud(paths_to_pointclouds);
//  gvl->getMap("mySecondRobotMap")->insertMetaPointCloud(myBinvoxCloud, eVT_OCCUPIED);


  //myRobotCloud.syncToDevice();
  //myRobotCloud.debugPointCloud();
  //gvl->getMap("mySecondRobotMap")->insertMetaPointCloud(myRobotCloud, eVT_SWEPT_VOLUME_END);

  // ============ END TESTING =============

  /*!
   * Now we enter "normal" operation
   */
  gpu_voxels::Matrix4f new_base_pose;
  new_base_pose.a11 = 1;
  new_base_pose.a22 = 1;
  new_base_pose.a33 = 1;
  new_base_pose.a44 = 1;

  new_base_pose.a14 += 0;
  new_base_pose.a24 += 2;
  new_base_pose.a34 += 0.5;

  std::vector<float> myRobotJointValues(rob_dim, 0.0);

  // start the Kinect
  kinect->run();

  //LOGGING_INFO(Gpu_voxels, "Updating robot part different size..." << endl);
  //std::vector<Vector3f> test_frame(123*234, 0.0);
  //gvl->updateRobotPart("myRobot", rob_dim-1, test_frame);


  // initialize the joint interpolation
  std::vector<float> min_joint_values(rob_dim, -1.0);
  std::vector<float> max_joint_values(rob_dim, 1.5);
  std::size_t counter = 0;
  const float ratio_delta = 0.02;
  const int num_swept_volumes = 50;//voxelmap::BIT_VECTOR_LENGTH;

  /*!
   * The robot moves and changes it's pose, so we "voxelize"
   * the links in every step and update the robot map.
   */

  LOGGING_INFO(Gpu_voxels, "Updating robot pose..." << endl);
  for (int i = 0; i < num_swept_volumes; ++i)
  {
    myRobotJointValues = gpu_voxels::CudaMath::interpolateLinear(min_joint_values, max_joint_values,
                                                                 ratio_delta * counter++);
    //gvl->updateRobotPart("myRobot", rob_dim-1, kinect->getDataPtr());
    new_base_pose.a14 += 0.05;
    new_base_pose.a24 += 0.05;
    new_base_pose.a34 += 0.00;

    gvl->updateRobotPose("myRobot", myRobotJointValues, &new_base_pose);
    VoxelType v = VoxelType(eVT_SWEPT_VOLUME_START + 1 + i);
//    if(i == 30)
//      v = eVT_COLLISION;
    gvl->insertRobotIntoMap("myRobot", "myRobotMap", v);
  }

  while (true)
  {
    /*!
     * While the robot moves, we have to update its cam pose:
     */
//    vector6f myCamPose();
//    gvl->updateCameraPose("myEnvironmentMap", myCamPose);
    /*!
     * Query current resolution level of the visualization.
     * Use this level for collision detection with specific resolution.
     */
//    uint32_t resolution_level = gvl->getVisualization("myEnvironmentMap")->getResolutionLevel();

    /*!
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
//    LOGGING_INFO(
//        Gpu_voxels,
//        "Collsions: "
//            << gvl->getMap("myEnvironmentMap")->collideWithResolution(gvl->getMap("myRobotMap"), 1.0f,
//                                                                      resolution_level) << endl);

    std::vector<float> joints(1, 0.0);
    gpu_voxels::Matrix4f kinect_base_pose;

    kinect_base_pose.a11 = 1;
    kinect_base_pose.a23 = 1;
    kinect_base_pose.a32 = -1;
    kinect_base_pose.a44 = 1;

    kinect_base_pose.a41 = 1;
    kinect_base_pose.a42 = 1;
    gvl->updateRobotPose("kinectData", joints, &kinect_base_pose);
    gvl->updateRobotPart("kinectData", 0, kinect->getDataPtr());
    gvl->insertRobotIntoMap("kinectData", "myEnvironmentMap", eVT_OCCUPIED);


   // gvl->getMap("myEnvironmentMap")->insertGlobalData(kinect->getDataPtr(), eVT_OCCUPIED);

    voxelmap::BitVectorVoxel collision_types;
    LOGGING_INFO(
        Gpu_voxels,
        "Collsions: "
            << gvl->getMap("myEnvironmentMap")->collideWithTypes(gvl->getMap("myRobotMap"), collision_types,
                                                                 1.0f) << endl);

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

    gvl->getVisualization("myRobotMap")->setDrawTypes(draw_types);


    // visualize both maps
    gvl->visualizeMap("myRobotMap");
    //gvl->visualizeMap("myPlanningMap");
    gvl->visualizeMap("myEnvironmentMap");
    //gvl->visualizeMap("mySecondRobotMap");

    usleep(10000);

    // We assume that the robot will be updated in the next loop
   // gvl->clearMap("myRobotMap");
    gvl->clearMap("myEnvironmentMap");
  }

}
