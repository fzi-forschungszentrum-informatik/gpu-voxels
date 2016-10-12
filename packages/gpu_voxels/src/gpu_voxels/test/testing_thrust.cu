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
 * \author  Herbert Pietrzyk <pietrzyk@fzi.de>
 * \date    2016-12-10
 *
 */
//----------------------------------------------------------------------

#include <cstdlib>
#include <thrust/binary_search.h>
#include <thrust/generate.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <boost/test/unit_test.hpp>
#include <gpu_voxels/test/testing_fixtures.hpp>
#include "icl_core_performance_monitor/PerformanceMonitor.h"


BOOST_FIXTURE_TEST_SUITE(thrustPerformance, ArgsFixture)

BOOST_AUTO_TEST_CASE(binarySearchComparison)
{
  thrust::host_vector<int> large_vec_h(10000000);
  srand(13);
  thrust::generate(large_vec_h.begin(), large_vec_h.end(), rand);

  thrust::device_vector<int> large_vec_d = large_vec_h;

  thrust::host_vector<int> small_vec_h(1000);
  srand(13);
  thrust::generate(small_vec_h.begin(), small_vec_h.end(), rand);

  thrust::device_vector<int> small_vec_d = small_vec_h;

  thrust::device_vector<bool> output_d(10000000);

  PERF_MON_START("binarySearchSmallVectorInLargeVector");
  for(int i = 0; i < iterationCount; i++)
  {

    thrust::binary_search(thrust::device,
                          large_vec_d.begin(), large_vec_d.end(),
                          small_vec_d.begin(), small_vec_d.end(),
                          output_d.begin());

    BOOST_CHECK_MESSAGE(true, "binarySearch 'Small in Large' finished");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("binarySearchSmallVectorInLargeVector", "binarySearchSmallVectorInLargeVector", "thrustPerformance");
  }
  
  PERF_MON_START("binarySearchLargeVectorInSmallVector");
  for(int i = 0; i < iterationCount; i++)
  {

    thrust::binary_search(thrust::device,
                          small_vec_d.begin(), small_vec_d.end(),
                          large_vec_d.begin(), large_vec_d.end(),
                          output_d.begin());

    BOOST_CHECK_MESSAGE(true, "binarySearch 'Large in Small' finished");
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("binarySearchLargeVectorInSmallVector", "binarySearchLargeVectorInSmallVector", "thrustPerformance");
  }
}

BOOST_AUTO_TEST_SUITE_END()
