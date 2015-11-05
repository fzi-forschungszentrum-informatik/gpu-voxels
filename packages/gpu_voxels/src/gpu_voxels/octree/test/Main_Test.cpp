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
 * \author  Florian Drews
 * \date    2014-01-08
 *
 */
//----------------------------------------------------------------------

#include "Tests.h"
#include "ArgumentHandling.h"

#include <vector_types.h>
#include <malloc.h>
#include <vector>
#include <gpu_voxels/helpers/cuda_datatypes.h>

using namespace gpu_voxels;

//class N
//{
//  uint8_t status;
//};
//
//class L : public virtual N
//{
//
//};
//
//class I : public virtual N
//{
//  uint8_t f;
//};
//
//class NP : public virtual N
//{
//  uint8_t p;
//};
//
//class LP : public NP, public L
//{
//
//};
//
//class IP : public NP, public I
//{
//
//};

// without virtual inheritance:
//N 1
//L 1
//I 2
//NP 1
//LP 2
//IP 3

// with virtual inheritance:
//N 1
//L 16
//I 16
//NP 16
//LP 32
//IP 32

//int main(int argc, char **argv)
//{
////  printf("N %u\n", sizeof(N));
////  printf("L %u\n", sizeof(L));
////  printf("I %u\n", sizeof(I));
////  printf("NP %u\n", sizeof(NP));
////  printf("LP %u\n", sizeof(LP));
////  printf("IP %u\n", sizeof(IP));
////  return 0;

//  uint32_t num_points;
//  std::vector<Vector3f> points;
//  std::string pcd_file = "./pointcloud_0002.pcd";
//  if(NTree::Test::readPcFile(pcd_file, points, num_points))
//  {
//    NTree::Test::run(points, num_points);
//  }else{
//    std::cout << "Error in reading test data from pointcloud" << std::endl;
//  }
//  return 0;
//}
