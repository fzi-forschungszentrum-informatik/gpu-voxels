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
 * \date    2016-06-05
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/helpers/PointCloud.h>
#include <gpu_voxels/test/testing_fixtures.hpp>
#include <boost/test/unit_test.hpp>

using namespace gpu_voxels;



BOOST_FIXTURE_TEST_SUITE(pointclouds, ArgsFixture)

BOOST_AUTO_TEST_CASE(meta_pointcloud_equality)
{
  PERF_MON_START("meta_pointcloud_equality");
  for(int i = 0; i < iterationCount; i++)
  {
    std::cout << getDeviceMemoryInfo();

    MetaPointCloud *orig = new MetaPointCloud();

    std::vector<Vector3f> testdata;

    for(size_t i = 0; i < (size_t)numberOfPoints; i++)
    {
      testdata.push_back(Vector3f(i, 0, 1.0/i));
    }
    for(size_t j = 0; j < 15; j++)
    {
      orig->addCloud(testdata);
    }
    orig->syncToDevice();

    MetaPointCloud *working_copy = new MetaPointCloud(*orig);

    for(size_t k = 0; k < 2000; k++)
    {
      MetaPointCloud* copy1 = new MetaPointCloud(*working_copy);
      MetaPointCloud* copy2 = new MetaPointCloud(*copy1);
      delete copy1;
      *working_copy = *copy2;
      delete copy2;
    }
    BOOST_CHECK_MESSAGE(*orig == *working_copy, "Point clouds are the same.");

    testdata[100] = Vector3f(100, 0.33, 1.0/100);
    working_copy->updatePointCloud(3, testdata, true);
    BOOST_CHECK_MESSAGE(!(*orig == *working_copy), "Point cloud difference detected.");

    delete orig;
    delete working_copy;

    std::cout << getDeviceMemoryInfo();
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("meta_pointcloud_equality", "meta_pointcloud_equality", "pointclouds");
  }
}

BOOST_AUTO_TEST_CASE(pointcloud_equality)
{
  PERF_MON_START("pointcloud_equality");
  for(int i = 0; i < iterationCount; i++)
  {
    std::cout << getDeviceMemoryInfo();

    PointCloud *orig = new PointCloud();

    std::vector<Vector3f> testdata;

    for(size_t i = 0; i < (size_t)numberOfPoints; i++)
    {
      testdata.push_back(Vector3f(i, 0, 1.0/i));
    }
    orig->update(testdata);

    PointCloud *working_copy = new PointCloud(*orig);

    for(size_t k = 0; k < 2000; k++)
    {
      PointCloud* copy1 = new PointCloud(*working_copy);
      PointCloud* copy2 = new PointCloud(*copy1);
      delete copy1;
      *working_copy = *copy2;
      delete copy2;
    }
    BOOST_CHECK_MESSAGE(*orig == *working_copy, "Point clouds are the same.");

    testdata[100] = Vector3f(100, 0.33, 1.0/100);
    working_copy->update(testdata);
    BOOST_CHECK_MESSAGE(!(*orig == *working_copy), "Point cloud difference detected.");

    delete orig;
    delete working_copy;

    std::cout << getDeviceMemoryInfo();
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("pointcloud_equality", "pointcloud_equality", "pointclouds");
  }
}


BOOST_AUTO_TEST_CASE(pointcloud_rotation)
{
  // This only tests the API. The math behind this is tested in testing_cuda_math.cu

  PERF_MON_START("pointcloud_rotation");
  for(int i = 0; i < iterationCount; i++)
  {

    PointCloud orig;
    PointCloud rotated;

    std::vector<Vector3f> testdata;
    for(size_t i = 0; i < (size_t)numberOfPoints; i++)
    {
      testdata.push_back(Vector3f(i, 0, 1.0/i));
    }
    orig.add(testdata);

    gpu_voxels::Matrix4f transformation = gpu_voxels::Matrix4f::createFromRotationAndTranslation(
          gpu_voxels::Matrix3f::createFromRPY(Vector3f(12.0, 11.0, 56.0)), Vector3f(2.2, 1.1, 3.3));


    orig.transform(&transformation, &rotated);
    orig.transformSelf(&transformation);

    BOOST_CHECK_MESSAGE(orig == rotated, "Self rotation and copy rotation are equal.");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("pointcloud_rotation", "pointcloud_rotation", "pointclouds");
  }
}

BOOST_AUTO_TEST_SUITE_END()

