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
 * \date    2014-11-27
 *
 * This example program demonstrates the performance gain when using an
 * tranform offset while checking for collisions, instead of transforming
 * one of the input pointclouds.
 *
 * 1. You can transform a pointcloud and insert later in a map
 * 2. The function "collideWith" allowes to set an offset.
 *
 * Since the calculation with an offset is accomplished on the gpu, we expect the latter attempt to be faster.
 *
 */
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>
#include <time.h>

#include <gpu_voxels/GpuVoxels.h>
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

  gvl = gpu_voxels::GpuVoxels::getInstance();
  gvl->initialize(100, 100, 100, 0.001); // ==> 100 Voxels, each one is 1 mm in size so the map represents 10x10x10 centimeter

  // add maps:
  gvl->addMap(MT_PROBAB_VOXELMAP, "myObjectVoxelmap");
  gvl->addMap(MT_BITVECTOR_VOXELLIST, "myObjectVoxellist");



  // load Voxellist
  gvl->insertPointCloudFromFile("myObjectVoxellist", "/schunk_svh/f20.binvox", true,
                                eBVM_OCCUPIED, true, Vector3f(0.06, 0.05, 0.01), 1.0);
  std::cout << "List generated" << std::endl;


  // load Metapointcloud to put into Voxelmap
  std::vector<Vector3f> pc;
  file_handling::PointcloudFileHandler::Instance()->loadPointCloud(
           "/schunk_svh/f20.binvox", true, pc, true, Vector3f(0.02, 0.05, 0.01), 1.0);
  std::cout << "Num points: " << pc.size() << std::endl;
  std::vector< std::vector<Vector3f> > vpc;
  vpc.push_back(pc);
  MetaPointCloud mpc(vpc);



  size_t colls;

  //----- insert map 100 times, transform, add to map and collide
  Matrix4f trans = Matrix4f::createFromRotationAndTranslation(
        Matrix3f::createIdentity(), Vector3f(0.0005, 0, 0));
  clock_t clock_tr_begin = clock();
  for (size_t i = 0; i < 100; i++)
  {
    gvl->getMap("myObjectVoxelmap")->clearMap();

    mpc.transformSelf(&trans);

    gvl->getMap("myObjectVoxelmap")->insertMetaPointCloud(mpc, eBVM_SWEPT_VOLUME_START);
    colls = gvl->getMap("myObjectVoxellist")->as<voxellist::BitVectorVoxelList>()->collideWith(gvl->getMap("myObjectVoxelmap")->as<voxelmap::ProbVoxelMap>(), 0.1);
    std::cout << "Transform then collide:\n colls in step " << i << " = " << colls << std::endl;
  }
  clock_t clock_tr_end = clock();



  // generate the same test setup for the second test
  gvl->getMap("myObjectVoxelmap")->clearMap();
  std::cout << "-------------------------------------" << std::endl;



  //----- insert map 100 times, calculate collision with offset
  Vector3i offset;
  clock_t clock_o_begin = clock();
  for (size_t i = 0; i < 100; i++)
  {
    gvl->getMap("myObjectVoxelmap")->clearMap();

    offset.x = i;
    gvl->getMap("myObjectVoxelmap")->insertMetaPointCloud(mpc, eBVM_SWEPT_VOLUME_START);
    colls = gvl->getMap("myObjectVoxellist")->as<voxellist::BitVectorVoxelList>()->collideWith(gvl->getMap("myObjectVoxelmap")->as<voxelmap::ProbVoxelMap>(), 0.1, offset);
    std::cout << "Collide with offset:\n colls in step " << i << " = " << colls << std::endl;
  }
  clock_t clock_o_end = clock();




  // compare timings
  double time_transform = (double) (clock_tr_end - clock_tr_begin) / CLOCKS_PER_SEC;;
  double time_offset = (double) (clock_o_end - clock_o_begin) / CLOCKS_PER_SEC;


  while(true)
  {
    gvl->visualizeMap("myObjectVoxelmap");
    gvl->visualizeMap("myObjectVoxellist");
    usleep(500000);
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "time with transform: " << time_transform << "\ntime with offset: " << time_offset << std::endl;
    std::cout << "-------------------------------------" << std::endl;

  }

}
