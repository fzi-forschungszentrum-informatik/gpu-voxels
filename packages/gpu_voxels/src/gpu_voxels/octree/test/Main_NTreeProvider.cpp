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
 * \date    2014-04-17
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/helpers/CompileIssues.h>

#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/test/Helper.h>
#include <gpu_voxels/octree/test/Kinect.h>
#include <gpu_voxels/octree/test/SensorData.h>

#include <vector_types.h>
#include <vector>
#include <gpu_voxels/octree/test/NTreeProvider.h>
#include <gpu_voxels/octree/test/Provider.h>
#include <gpu_voxels/octree/test/VoxelMapProvider.h>
#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/EnvNodesProbabilistic.h>
#include <gpu_voxels/octree/test/OctomapProvider.h>

#include <icl_core_performance_monitor/PerformanceMonitor.h>

#include <boost/thread/barrier.hpp>
#include <boost/thread.hpp>

#include <signal.h> // or <csignal> in C++

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>       /* time */
#include <math.h>

using namespace gpu_voxels::NTree;
using namespace gpu_voxels::NTree::Provider;

boost::barrier* my_barrier;

std::vector<boost::thread*> vec_threads;
std::vector<gpu_voxels::NTree::Provider::Provider*> vec_provider;
std::vector<SensorData*> vec_sensor_data;
volatile bool global_start;
volatile bool global_stop;

/**
 * Save handling of program exit
 */
void freeAll()
{
  bool second_call = global_stop == true;
  global_stop = true;
  bool all_done = true;
  for (uint32_t i = 0; i < vec_provider.size(); ++i)
  {
    printf("join\n");
    bool tmp = vec_threads[i]->timed_join(boost::posix_time::milliseconds(1000));
    all_done &= tmp;
    if(tmp)
      printf("join finished\n");
  }

  if(second_call || all_done)
  {
    for (uint32_t i = 0; i < vec_provider.size(); ++i)
    {
      if (vec_sensor_data[i] != NULL)
      {
        vec_sensor_data[i]->stop();
        delete vec_sensor_data[i];
        vec_sensor_data[i] = NULL;
      }
      if (vec_provider[i] != NULL)
      {
        delete vec_provider[i];
        vec_provider[i] = NULL;
      }
    }
    exit(0);
  }
}

void ctrlchandler(int)
{
  freeAll();
}
void killhandler(int)
{
  freeAll();
}

void thread_handleProvider(Provider_Parameter& parameter, uint32_t id)
{
  gpu_voxels::NTree::Provider::Provider** provider = &vec_provider[id];
  SensorData** kinect = &vec_sensor_data[id];
  *kinect = NULL;
  *provider = NULL;

  switch (parameter.type)
  {
    case Provider_Parameter::TYPE_OCTREE:
      *provider = new NTreeProvider();
      break;
    case Provider_Parameter::TYPE_VOXEL_MAP:
      *provider = new VoxelMapProvider();
      break;
    case Provider_Parameter::TYPE_OCTOMAP:
      *provider = new OctomapProvider();
  }

  (**provider).init(parameter);

  (**provider).visualize();

  switch (parameter.mode)
  {
    case Provider_Parameter::MODE_PTU_LIVE:
    case Provider_Parameter::MODE_KINECT_LIVE:
    case Provider_Parameter::MODE_KINECT_PLAYBACK:
      *kinect = new Kinect(*provider, &parameter);
      break;
    default:
      break;
  }

//  if (use_kinect)
//  {
//    *kinect = new Kinect(*provider, &parameter);
//  }

  my_barrier->wait();
  // output of main thread

  // set octree which should collide
  if (parameter.collide)
    (**provider).setCollideWith(vec_provider[id + 1]);

  my_barrier->wait();

  if (global_stop)
  {
    if (*kinect != NULL)
    {
      delete *kinect;
      *kinect = NULL;
    }
    return;
  }

  if (*kinect == NULL && parameter.collide)
    (**provider).collide();

  if (*kinect != NULL)
    (**kinect).run();

  while (!global_stop)
  {
#ifdef MANUAL_MODE
    std::cout << "Press to take picture" << std::endl;
    std::string exit = "";
    std::cin >> exit;
#ifdef MODE_KINECT
    (**kinect).takeImage();
#else
    usleep(uint32_t(1 / parameter.kinect_fps * 1000000));

    std::vector<Vector3f> points;
    uint32_t num_points = 0;
    readPCD(KINECT_FILE, points, num_points);

    provider->newKinectData(points, num_points);
    //break;
#endif
#else

    if (!(**provider).waitForNewData(&global_stop))
      break;

    if (*kinect == NULL && parameter.collide)
      (**provider).collide();

#endif

    (**provider).visualize();
  }

  if (*kinect != NULL)
  {
    (*kinect)->stop();
    delete *kinect;
    *kinect = NULL;
  }

  delete *provider;
  *provider = NULL;
}

static void printHelp()
{
  printf("\n\n##### Help for octree_provider #####\n");
  Provider_Parameter::printHelp();
  printf("Example:\n   octree_provider -shm 0 -m load_pc -f ./pointcloud.pcd \n");
  printf(
      "Example:\n   octree_provider -shm 0 -m kinect_live -fps 10 -shm 1 -m load_pc -f ./pointcloud.pcd \n");
}

int main(int argc, char **argv)
{
  signal(SIGINT, ctrlchandler);
  signal(SIGTERM, killhandler);

  icl_core::logging::initialize();

  srand(Test::RAND_SEED);
  srand48(Test::RAND_SEED);

  PERF_MON_ENABLE_ALL(true);


  printf("sizeof(Environment::InnerNode) = %lu\n", sizeof(Environment::InnerNode));
  printf("sizeof(Environment::LeafNode) = %lu\n", sizeof(Environment::LeafNode));
  printf("sizeof(Environment::NodeData) = %lu\n", sizeof(Environment::NodeData));
  printf("sizeof(Environment::InnerNodeProb) = %lu\n", sizeof(Environment::InnerNodeProb));
  printf("sizeof(Environment::LeafNodeProb) = %lu\n", sizeof(Environment::LeafNodeProb));
  printf("sizeof(Environment::NodeDataProb) = %lu\n", sizeof(Environment::NodeDataProb));

  std::vector<Provider_Parameter> parameter;
  bool error = parseArguments(parameter, argc, argv);
  if (error || argc <= 1)
  {
    printHelp();
    return 0;
  }
  printf("num parameter: %lu\n", parameter.size());
  if(!readPcFile(parameter))
  {
    printf("Error reading pointcloud file!\n");
    return 0;
  }

  Test::getRandomPlans(parameter);

  Test::testAndInitDevice();

  my_barrier = new boost::barrier(parameter.size() + 1);
  global_stop = false;
  vec_threads.resize(parameter.size());
  vec_sensor_data.resize(parameter.size());
  vec_provider.resize(parameter.size());
  for (uint32_t i = 0; i < parameter.size(); ++i)
  {
    vec_sensor_data[i] = NULL;
    vec_provider[i] = NULL;
    vec_threads[i] = new boost::thread(thread_handleProvider, parameter[i], i);
  }
  my_barrier->wait();

//  std::cout << "Press any key to start" << std::endl;
//  std::string exit = "";
//  std::cin >> exit;

  my_barrier->wait();

  std::cout << "STARTED!!" << std::endl;

  while (true)
  {
    std::cout << "Input 'exit' to stop" << std::endl;
    std::string exit = "";
    std::cin >> exit;
    if (exit == "exit")
      break;
  }

  global_stop = true;

  // wait till deconstruction work is done
  while(true)
  {
    bool all_done = true;
    for (uint32_t i = 0; i < vec_provider.size(); ++i)
    {
      printf("join\n");
      bool tmp = vec_threads[i]->timed_join(boost::posix_time::milliseconds(1000));
      all_done &= tmp;
      if(tmp)
        printf("join finished\n");
    }
    if(all_done)
      break;
  }
  freeAll();
  return 0;
}

