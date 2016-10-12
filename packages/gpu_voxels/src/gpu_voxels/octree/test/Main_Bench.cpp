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
 * \date    2014-05-20
 *
 */
//----------------------------------------------------------------------

#include "ArgumentHandling.h"
#include "SensorData.h"
#include "Provider.h"
//#include "Bench.h"
#include "Kinect.h"
#include "NTreeProvider.h"
#include "VoxelMapProvider.h"
#include "OctomapProvider.h"

#include <icl_core_performance_monitor/PerformanceMonitor.h>
#include "Helper.h"

using namespace gpu_voxels::NTree;
using namespace gpu_voxels::NTree::Bench;
using namespace gpu_voxels::NTree::Provider;

Bech_Parameter parameter;
std::vector<gpu_voxels::NTree::Provider::Provider*> provider;
const int num_names = 1000, num_events = 1000;
SensorData* sensor_data = NULL;

void build()
{
  bool build_mode = false;
  int runs = 1;
  if (parameter.mode == Bech_Parameter::MODE_BUILD)
  {
    build_mode = true;
    runs = parameter.runs;
    PERF_MON_ENABLE("build");
    PERF_MON_ENABLE("propagate");
    PERF_MON_ENABLE("lb_propagate");
    PERF_MON_ENABLE("Octomap::init");

    for (int res = parameter.resolution_from; res <= parameter.resolution_to;
        res = ceil(res * parameter.resolution_scaling))
    {
      for (int b = parameter.blocks_from; b <= parameter.blocks_to; b += parameter.blocks_step)
      {

        for (int t = parameter.threads_from; t <= parameter.threads_to; t += parameter.threads_step)
        {
          PERF_MON_ADD_STATIC_DATA_P("BLOCKS", b, "build");
          PERF_MON_ADD_STATIC_DATA_P("THREADS", t, "build");

          for (int r = 0; r < runs; ++r)
          {
            for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
            {
              Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
              gpu_voxels::NTree::Provider::Provider* m_provider = NULL;

              // free data of last iteration
              if (provider[i] != NULL)
                delete provider[i];

              switch (my_parameter->type)
              {
                case Provider_Parameter::TYPE_OCTREE:
                  m_provider = new NTreeProvider();
                  break;
                case Provider_Parameter::TYPE_VOXEL_MAP:
                  m_provider = new VoxelMapProvider();
                  break;
                case Provider_Parameter::TYPE_OCTOMAP:
                  m_provider = new OctomapProvider();
              }
              provider[i] = m_provider;
            }

            // init
            for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
            {
              Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
              uint32_t tmp_res = my_parameter->resolution_tree;
              if (build_mode)
              {
                my_parameter->resolution_tree = res;
                my_parameter->num_blocks = b;
                my_parameter->num_threads = t;
              }

              PERF_MON_ADD_STATIC_DATA_P("RESOLUTION", my_parameter->resolution_tree, "build");
              provider[i]->init(*my_parameter);

              if (build_mode)
                my_parameter->resolution_tree = tmp_res;
            }
            if (!build_mode)
              break;
          }

          if (build_mode)
          {
            // performance logging

            PERF_MON_SUMMARY_ALL_INFO;
            PERF_MON_INITIALIZE(num_names, num_events);
          }
          else
            break;
        }
        if (!build_mode)
          break;
      }
      if (!build_mode)
        break;
    }
  }
}

void collideRun()
{
  // generate new robot plans
  PERF_MON_ENABLE("build");
  std::vector<std::vector<gpu_voxels::Vector3f> > points_backup;
  for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
  {
    Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
    gpu_voxels::NTree::Provider::Provider* m_provider = NULL;

    if (my_parameter->mode == Provider_Parameter::MODE_RANDOM_PLAN)
    {
      // free data of last iteration
      if (provider[i] != NULL)
        delete provider[i];

      switch (my_parameter->type)
      {
        case Provider_Parameter::TYPE_OCTREE:
          m_provider = new NTreeProvider();
          break;
        case Provider_Parameter::TYPE_VOXEL_MAP:
          m_provider = new VoxelMapProvider();
          break;
        case Provider_Parameter::TYPE_OCTOMAP:
          m_provider = new OctomapProvider();
      }
      provider[i] = m_provider;

      std::vector<gpu_voxels::Vector3f> rand_plan;
      Test::getRandomPlan(my_parameter->points, rand_plan, 5, my_parameter->plan_size);
      points_backup.push_back(my_parameter->points);
      my_parameter->points.swap(rand_plan);

      PERF_MON_ADD_STATIC_DATA_P("NumPlanPoints", my_parameter->points.size(), "collideRun");

      provider[i]->init(*my_parameter);
    }
  }
  PERF_MON_DISABLE("build");

  // replays of collide with same robot plan
  for (int w = 0; w < parameter.replay; ++w)
  {
    for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
    {
      Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
      gpu_voxels::NTree::Provider::Provider* m_provider = provider[i];
      if (my_parameter->collide)
        m_provider->setCollideWith(provider[i + 1]);

      m_provider->collide();
    }
  }

  // restore robot points
  for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
  {
    Provider_Parameter* my_parameter = &parameter.provider_parameter[i];

    if (my_parameter->mode == Provider_Parameter::MODE_RANDOM_PLAN)
    {
      my_parameter->points.swap(points_backup.front());
      points_backup.erase(points_backup.begin());
    }
  }
}

void insert_collide()
{
  bool run = false;

  if (parameter.mode == Bech_Parameter::MODE_INSERT)
  {
    run = true;
  }
  else if (parameter.mode == Bech_Parameter::MODE_COLLIDE_LIVE || parameter.mode == Bech_Parameter::MODE_COLLIDE)
  {
    run = true;
    PERF_MON_ENABLE("collideRun");
    PERF_MON_ENABLE("collide_wo_locking");
    PERF_MON_ENABLE("intersect_load_balance");
    PERF_MON_ENABLE("VoxelMapProvider::collide_wo_locking");
    PERF_MON_ENABLE("VoxelMap::intersect_load_balance");
    PERF_MON_ENABLE("intersect_sparse");
  }

  if (run)
  {
    PERF_MON_ENABLE("insert");
    PERF_MON_ENABLE("newSensorData");
    PERF_MON_ENABLE("insertVoxel");
    PERF_MON_ENABLE("packVoxel_Map");
    PERF_MON_ENABLE("computeFreeSpaceViaRayCast");
    PERF_MON_ENABLE("processSensorData");
    PERF_MON_ENABLE("transformKinectPointCloud_simple");
    PERF_MON_ENABLE("lb_propagate");
    PERF_MON_ENABLE("propagate");
    //PERF_MON_ENABLE("rebuild"); // TODO: handle case where so rebuild occurs but data is needed for correct column alignment
    PERF_MON_ENABLE("Octomap::newSensorData");

    int runs = parameter.runs;

    for (int res = parameter.resolution_from; res <= parameter.resolution_to;
        res = ceil(res * parameter.resolution_scaling))
    {
      for (int b = parameter.blocks_from; b <= parameter.blocks_to; b += parameter.blocks_step)
      {
        for (int t = parameter.threads_from; t <= parameter.threads_to; t += parameter.threads_step)
        {
          // use same seed for comparison of results for different resolutions etc.
          srand(Test::RAND_SEED);
          srand48(Test::RAND_SEED);

          // delete old objects, create new ones and init
          //usleep(100000);
          if (sensor_data != NULL)
          {
            delete sensor_data;
            sensor_data = NULL;
          }
          //usleep(100000);
          for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
          {
            Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
            gpu_voxels::NTree::Provider::Provider* m_provider = NULL;

            // free data of last iteration
            if (provider[i] != NULL)
              delete provider[i];

            switch (my_parameter->type)
            {
              case Provider_Parameter::TYPE_OCTREE:
                m_provider = new NTreeProvider();
                break;
              case Provider_Parameter::TYPE_VOXEL_MAP:
                m_provider = new VoxelMapProvider();
                break;
              case Provider_Parameter::TYPE_OCTOMAP:
                m_provider = new OctomapProvider();
            }
            provider[i] = m_provider;

            // set resolution for this iteration
            my_parameter->resolution_tree = my_parameter->resolution_free =
                my_parameter->resolution_occupied = (uint32_t) res;
            my_parameter->num_blocks = b;
            my_parameter->num_threads = t;

            my_parameter->save_collisions = my_parameter->clear_collisions = parameter.save_collisions;
            my_parameter->min_collision_level = 0;

            provider[i]->init(*my_parameter);
          }

          // init sensor (Kinect)
          for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
          {
            Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
            switch (my_parameter->mode)
            {
              case Provider_Parameter::MODE_PTU_LIVE:
              case Provider_Parameter::MODE_KINECT_LIVE:
              case Provider_Parameter::MODE_KINECT_PLAYBACK:
              {
                if (sensor_data != NULL)
                {
                  printf("ERROR only one sensor source (Kinect) allowed!\n");
                  exit(1);
                }
                sensor_data = new Kinect(provider[i], my_parameter);
                break;
              }
              default:
                break;
            }

            if (my_parameter->collide)
              provider[i]->setCollideWith(provider[i + 1]);
          }

          for (int clevel = parameter.collision_level_from; clevel <= parameter.collision_level_to; clevel +=
              parameter.collision_level_step)
          {
            PERF_MON_ADD_STATIC_DATA_P("BLOCKS", b, "insert");
            PERF_MON_ADD_STATIC_DATA_P("THREADS", t, "insert");
            PERF_MON_ADD_STATIC_DATA_P("CollisionLevel", clevel, "insert");
            PERF_MON_ADD_STATIC_DATA_P("RESOLUTION", res, "newSensorData");

            // start sensor
            if (sensor_data != NULL)
              sensor_data->run();

            // insert kinect data
            for (int r = 0; r < runs; ++r)
            {
              for (size_t i = 0; i < parameter.provider_parameter.size(); ++i)
              {
                Provider_Parameter* my_parameter = &parameter.provider_parameter[i];
                my_parameter->min_collision_level = clevel;
              }

              if (sensor_data != NULL)
                sensor_data->takeImage(); // waits till all providers' work is done

              if (parameter.mode == Bech_Parameter::MODE_COLLIDE)
              {
                collideRun();
              }

              // log every run or only after last one
              if (parameter.log_runs || r == runs - 1)
              {
                // performance logging
                PERF_MON_SUMMARY_ALL_INFO;
                PERF_MON_INITIALIZE(num_names, num_events);
                printf("Interation done\n");
              }
            }
            if (sensor_data != NULL)
              sensor_data->stop();
          }
        }
      }
    }
  }
}

void run()
{
  provider = std::vector<gpu_voxels::NTree::Provider::Provider*>(parameter.provider_parameter.size(), NULL);
//  for(size_t i = 0; i < provider.size(); ++i)
//    provider[i] = NULL;

  std::string m = "";
  if (parameter.mode == Bech_Parameter::MODE_BUILD)
    m = "BUILD";
  else if (parameter.mode == Bech_Parameter::MODE_INSERT)
    m = "INSERT";
  else if (parameter.mode == Bech_Parameter::MODE_COLLIDE_LIVE)
    m = "COLLIDE_LIVE";
  else if (parameter.mode == Bech_Parameter::MODE_COLLIDE)
    m = "COLLIDE";
  std::string t = getTime_str();
  std::string tree_type;
#ifdef PROBABILISTIC_TREE
  tree_type = "PROB";
#else
  tree_type = "DET";
#endif
  std::string filename = "./Benchmarks/" + m + "/" + getUname().nodename + "_" + tree_type + "_" + t + ".log";
  printf("Log file: %s\n", filename.c_str());
  ofstream log(filename.c_str());

  utsname uname = getUname();
  log << "############## HEADER ################" << std::endl;
  log << "#   " << parameter.command << std::endl;
  log << "#   " << std::endl;
  log << "#   " << "Domain name: " << uname.domainname << std::endl;
  log << "#   " << "Machine: " << uname.machine << std::endl;
  log << "#   " << "Node name: " << uname.nodename << std::endl;
  log << "#   " << "Release: " << uname.release << std::endl;
  log << "#   " << "Sys name: " << uname.sysname << std::endl;
  log << "#   " << "Version: " << uname.version << std::endl;
  log << "############## HEADER ################" << std::endl;
  log << std::endl;

  PERF_MON_INITIALIZE(num_names, num_events);

  build();

  insert_collide();

  log.close();
}

int main(int argc, char **argv)
{
//  signal(SIGINT, ctrlchandler);
//  signal(SIGTERM, killhandler);

  icl_core::logging::initialize();

  printf("sizeof(Environment::InnerNode) = %lu\n", sizeof(Environment::InnerNode));
  printf("sizeof(Environment::LeafNode) = %lu\n", sizeof(Environment::LeafNode));
  printf("sizeof(Environment::NodeData) = %lu\n", sizeof(Environment::NodeData));
  printf("sizeof(Environment::InnerNodeProb) = %lu\n", sizeof(Environment::InnerNodeProb));
  printf("sizeof(Environment::LeafNodeProb) = %lu\n", sizeof(Environment::LeafNodeProb));
  printf("sizeof(Environment::NodeDataProb) = %lu\n", sizeof(Environment::NodeDataProb));

#ifdef PROBABILISTIC_TREE
  printf("Using probabilistic NTree\n");
#else
  printf("Using deterministic NTree\n");
#endif

  bool error = parseArguments(parameter, argc, argv, true);
  if (error || argc <= 1)
  {
    //printHelp();
    return 0;
  }
  if(readPcFile(parameter.provider_parameter))
  {
    printf("Error reading pcd file!\n");
    return 0;
  }

  Test::testAndInitDevice();

  run();

  return 0;
}

