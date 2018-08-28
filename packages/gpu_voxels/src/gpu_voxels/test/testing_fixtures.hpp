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

#ifndef GPU_VOXELS_TEST_TESTING_FIXTURES_HPP_INCLUDED
#define GPU_VOXELS_TEST_TESTING_FIXTURES_HPP_INCLUDED

#define IC_PERFORMANCE_MONITOR
#include <iostream>
#include <fstream>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <gpu_voxels/helpers/cuda_handling.h>
#include "icl_core_performance_monitor/PerformanceMonitor.h"

#include <boost/version.hpp>
// boost::unit_test::traverse_test_tree is only part of public boost api since 1.59
#if BOOST_VERSION >= 105900
#include <boost/test/tree/traverse.hpp>
#endif
#include <boost/test/unit_test.hpp>

struct Visitor : boost::unit_test::test_tree_visitor
{
  void visit(boost::unit_test::test_case const& test)
  {  }

  bool test_suite_start(boost::unit_test::test_suite const& suite)
  {
    PERF_MON_ENABLE(suite.p_name);
    return true;
  }

  void test_suite_finish(boost::unit_test::test_suite const& suite)
  {

  }
};

struct GlobalFixture {

  boost::program_options::variables_map vm;

  GlobalFixture()
  {
    // Put together a timestamped filename:
    time_t now;
    char filename[100];
    filename[0] = '\0';
    now = time(NULL);
    if (now != -1)
    {
      strftime(filename, 100, "GPUVoxelsBenchmarkProtocol_%F_%H_%M.txt", gmtime(&now));
    }

    //parse Protocol store directory from command line
    boost::program_options::options_description desc("Test Parameters");
    desc.add_options()
        ("helpTest,H", "produce help message")
        ("outputPath,O", boost::program_options::value<std::string>()->default_value(filename), "file path to store protocol to")
        ("iterationCount,I", boost::program_options::value<int>()->default_value(1), "number of Iterations for each testcase")
        ("dimX,X", boost::program_options::value<int>()->default_value(89), "X Dimension of all maps")
        ("dimY,Y", boost::program_options::value<int>()->default_value(123), "Y Dimension of all maps")
        ("dimZ,Z", boost::program_options::value<int>()->default_value(74), "Z Dimension of all maps")
        ("numberOfPoints,N", boost::program_options::value<int>()->default_value(10000), "Number of points to be used during collision")
    ;

    int argc = boost::unit_test::framework::master_test_suite().argc;
    char** argv = boost::unit_test::framework::master_test_suite().argv;

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("helpTest")) {
        std::cout << desc << "\n";
    }

    Visitor visitor;
    boost::unit_test::traverse_test_tree(boost::unit_test::framework::master_test_suite(), visitor);

    std::cout << "===========================================================" << std::endl;
    std::cout << "=      Please run this Benchmark at least two times       =" << std::endl;
    std::cout << "=   to avoid measuring just in time compilation effects   =" << std::endl;
    std::cout << "===========================================================" << std::endl;

    PERF_MON_INITIALIZE(30, vm["iterationCount"].as<int>());
    PERF_MON_ENABLE("total");
    PERF_MON_START("runtime");

  }

  ~GlobalFixture()
  {
    PERF_MON_SILENT_MEASURE_AND_RESET_INFO_P("runtime", "runtime", "total");

    //PERF_MON_SUMMARY_ALL_INFO;
    ::icl_core::perf_mon::PerformanceMonitor* mon = ::icl_core::perf_mon::PerformanceMonitor::getInstance();
    std::stringstream ss;
    ss << mon->printSummaryAll();

    std::string outputPath = vm["outputPath"].as<std::string>();

    std::ofstream file;
    file.open(outputPath.c_str());
    if(!file.is_open())
    {
      std::cout << "Error opening log file " << outputPath << std::endl;
    }else{
      std::cout << "Opened log file " << outputPath << std::endl;
    }
    file << gpu_voxels::getDeviceInfos();
    file << "Results: " << std::endl << ss.str();
    file.close();
  }
};

struct ArgsFixture {

  int iterationCount;
  int dimX;
  int dimY;
  int dimZ;
  int numberOfPoints;

  ArgsFixture()
  {
    boost::program_options::options_description desc("Test Parameters");
    desc.add_options()
        ("helpTest,H", "produce help message")
        ("outputPath,O", boost::program_options::value<std::string>()->default_value("BenchmarkProtocol.txt"), "file path to store protocol to")
        ("iterationCount,I", boost::program_options::value<int>()->default_value(1), "number of Iterations for each testcase")
        ("dimX,X", boost::program_options::value<int>()->default_value(89), "X Dimension of all maps")
        ("dimY,Y", boost::program_options::value<int>()->default_value(123), "Y Dimension of all maps")
        ("dimZ,Z", boost::program_options::value<int>()->default_value(74), "Z Dimension of all maps")
        ("numberOfPoints,N", boost::program_options::value<int>()->default_value(10000), "Number of points to be used during collision")
    ;

    int argc = boost::unit_test::framework::master_test_suite().argc;
    char** argv = boost::unit_test::framework::master_test_suite().argv;


    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    iterationCount = vm["iterationCount"].as<int>();
    dimX = vm["dimX"].as<int>();
    dimY = vm["dimY"].as<int>();
    dimZ = vm["dimZ"].as<int>();
    numberOfPoints = vm["numberOfPoints"].as<int>();

  }

  ~ArgsFixture()
  {

  }

};

#endif // GPU_VOXELS_TEST_TESTING_FIXTURES_HPP_INCLUDED

