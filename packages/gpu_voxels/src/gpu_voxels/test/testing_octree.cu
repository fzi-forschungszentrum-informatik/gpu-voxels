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
 * \date    2015-10-14
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/octree/test/Tests.h>
#include <gpu_voxels/helpers/common_defines.h>
#include "gpu_voxels/helpers/PointcloudFileHandler.h"
#include <gpu_voxels/test/testing_fixtures.hpp>
#include <boost/test/unit_test.hpp>

using namespace gpu_voxels;



BOOST_FIXTURE_TEST_SUITE(octree_selftest, ArgsFixture)


BOOST_AUTO_TEST_CASE(morton_code_calculation)
{
  PERF_MON_START("morton_code_calculation");
  for(int i = 0; i < iterationCount; i++)
  {
    BOOST_CHECK_MESSAGE(NTree::Test::mortonTest(9998), "Morton Code Calculation");
  }
  PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("morton_code_calculation", "morton_code_calculation", "octree_selftest");
}


BOOST_AUTO_TEST_CASE(insert_voxels)
{
  PERF_MON_START("insert_voxels");
  for(int i = 0; i < iterationCount; i++)
  {
    uint32_t num_build = 16541;
    uint32_t num_insert = 26435;

    BOOST_CHECK_MESSAGE(NTree::Test::insertTest(num_build, num_insert, false, true), "Insert Voxels: Not Set Free, Propagate up.");
    BOOST_CHECK_MESSAGE(NTree::Test::insertTest(num_build, num_insert, true, true), "Insert Voxels: Set Free, Propagate up.");
    // These must fail:
    BOOST_CHECK_MESSAGE(!NTree::Test::insertTest(num_build, num_insert, true, false), "Negative Test: Insert Voxels: Set Free, No Propagate up");
    BOOST_CHECK_MESSAGE(!NTree::Test::insertTest(num_build, num_insert, false, false), "Negative Test: Insert Voxels: Not Set Free, No Propagate Up");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("insert_voxels", "insert_voxels", "octree_selftest");
  }
}



BOOST_AUTO_TEST_CASE(build_and_rebuild)
{
  PERF_MON_START("build_and_rebuild");
  for(int i = 0; i < iterationCount; i++)
  {
    BOOST_CHECK_MESSAGE(NTree::Test::testAndInitDevice(), "Device Capabilities of GPU");

    std::vector<Vector3f> points;
    std::vector<Vector3f> no_points;
    std::string pcd_file_name = "pointcloud_0002.pcd";
    BOOST_CHECK_MESSAGE(file_handling::PointcloudFileHandler::Instance()->loadPointCloud(pcd_file_name, true, points),
                        "Read pointcloud file");

    BOOST_CHECK_MESSAGE((points.size() > 0), "Pointcloud contains points");

    double time = 0;
    BOOST_CHECK_MESSAGE(NTree::Test::buildTest(points, points.size(), time, true), "Build Octree from pointcloud with Rebuild");
    BOOST_CHECK_MESSAGE(NTree::Test::buildTest(points, points.size(), time, false), "Build Octree from pointcloud with No Rebuild");
    // These may fail in RELEASE builds...?
    BOOST_CHECK_MESSAGE(NTree::Test::buildTest(no_points, points.size(), time, true), "Build Octree from empty pointcloud with Rebuild");
    BOOST_CHECK_MESSAGE(NTree::Test::buildTest(no_points, points.size(), time, false), "Build Octree from empty pointcloud with No Rebuild");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("build_and_rebuild", "build_and_rebuild", "octree_selftest");
  }
}


BOOST_AUTO_TEST_SUITE_END()


