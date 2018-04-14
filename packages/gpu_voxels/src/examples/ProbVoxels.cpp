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
* \author  Andreas Hermann <hermann@fzi.de>
* \date    2018-04-01
*
* This program demonstrates how work with probabilistic Voxels.
*
*/
//----------------------------------------------------------------------
#include <cstdlib>
#include <signal.h>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/GeometryGeneration.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

using namespace gpu_voxels;
using namespace voxelmap;
using namespace geometry_generation;

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

  gvl = GpuVoxels::getInstance();

  Vector3ui dim(136, 136, 136);
  float side_length = 1.0; // voxel side length
  gvl->initialize(dim.x, dim.y, dim.z, side_length);

  gvl->addMap(MT_PROBAB_VOXELMAP, "myProbVoxelMap");
  boost::shared_ptr<ProbVoxelMap> prob_map(gvl->getMap("myProbVoxelMap")->as<ProbVoxelMap>());

  std::vector<Vector3f> boxpoints;
  boxpoints = createBoxOfPoints( Vector3f(10, 10, 30), Vector3f(30, 30, 50), 0.9); // choose large delta, so that only 1 point falls into each voxel (mostly)


  PointCloud box(boxpoints);
  Matrix4f shift_diag_up(Matrix4f::createFromRotationAndTranslation(Matrix3f::createIdentity(), Vector3f(0, 15, 15)));
  Matrix4f shift_down(Matrix4f::createFromRotationAndTranslation(Matrix3f::createIdentity(), Vector3f(0, 15, -15)));
  Matrix4f shift_diag_down(Matrix4f::createFromRotationAndTranslation(Matrix3f::createIdentity(), Vector3f(0, -15, -15)));


  // insert cube into map
  prob_map->insertPointCloud(box, eBVM_MAX_OCC_PROB); // = 254
  box.transformSelf(&shift_diag_up);
  prob_map->insertPointCloud(box, eBVM_MAX_OCC_PROB); // = 254
  box.transformSelf(&shift_diag_up);
  prob_map->insertPointCloud(box, BitVoxelMeaning(229));
  box.transformSelf(&shift_diag_up);
  prob_map->insertPointCloud(box, BitVoxelMeaning(204));
  box.transformSelf(&shift_diag_up);
  prob_map->insertPointCloud(box, BitVoxelMeaning(179));
  box.transformSelf(&shift_diag_up);
  prob_map->insertPointCloud(box, BitVoxelMeaning(154));
  box.transformSelf(&shift_down);

                // eBVM_UNCERTAIN_OCC_PROB); // = 129

  prob_map->insertPointCloud(box, BitVoxelMeaning(104));
  box.transformSelf(&shift_diag_down);
  prob_map->insertPointCloud(box, BitVoxelMeaning(79));
  box.transformSelf(&shift_diag_down);
  prob_map->insertPointCloud(box, BitVoxelMeaning(54));
  box.transformSelf(&shift_diag_down);
  prob_map->insertPointCloud(box, BitVoxelMeaning(29));
  box.transformSelf(&shift_diag_down);
  prob_map->insertPointCloud(box, eBVM_MAX_FREE_PROB); // = 4
  box.transformSelf(&shift_diag_down);
  prob_map->insertPointCloud(box, eBVM_MAX_FREE_PROB); // = 4


  boxpoints = createBoxOfPoints( Vector3f(10, 5, 5), Vector3f(12, 130, 130), 0.9); // choose large delta, so that only 1 point falls into each voxel (mostly)
  box = PointCloud(boxpoints);
  prob_map->insertPointCloud(box, eBVM_UNCERTAIN_OCC_PROB); // = 129
  // this will not influence voxels which were also set to other probabilities, as it converts to adding 0 to their values.

  while(true)
  {
    gvl->visualizeMap("myProbVoxelMap");

    // this will show partly overlapping cubes, whose occupancy values will get summed up.
    // all "unknown" voxels will not get drawn, but the uncertain ones will (unkown = initialization value, uncertain = 0.5 probability)

    sleep(1);
  }

  return 0;
}
