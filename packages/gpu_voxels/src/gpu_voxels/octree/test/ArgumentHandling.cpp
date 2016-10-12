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
 * \date    2014-03-24
 *
 */
//----------------------------------------------------------------------

#include "ArgumentHandling.h"
#include "gpu_voxels/helpers/PointcloudFileHandler.h"

using namespace std;

namespace gpu_voxels {
namespace NTree {

namespace Benchmark {
void parseArguments(Benchmark_Parameter& parameter, int argc, char **argv)
{
  for (int i = 0; i < argc; ++i)
  {
    parameter.command += argv[i];
    parameter.command += " ";
  }

  for (int i = 0; i < argc; ++i)
  {
    std::string a = argv[i];
    if (a.compare("-m") == 0)
    {
      a = argv[++i];
      if (a.compare("FIND_CUBE") == 0)
        parameter.mode = Benchmark_Parameter::MODE_FIND_CUBE;
      else if (a.compare("INTERSECT_CUBE") == 0)
        parameter.mode = Benchmark_Parameter::MODE_INTERSECT_CUBE;
      else if (a.compare("INTERSECT_LINEAR") == 0)
        parameter.mode = Benchmark_Parameter::MODE_INTERSECT_LINEAR;
      else if (a.compare("BUILD_LINEAR") == 0)
        parameter.mode = Benchmark_Parameter::MODE_BUILD_LINEAR;
      else if (a.compare("BUILD_PCF") == 0)
        parameter.mode = Benchmark_Parameter::MODE_BUILD_PCF;
      else if (a.compare("SORT_PCF") == 0)
        parameter.mode = Benchmark_Parameter::MODE_SORT_PCF;
      else if (a.compare("SORT_LINEAR") == 0)
        parameter.mode = Benchmark_Parameter::MODE_SORT_LINEAR;
      else if (a.compare("INSERT_ROTATE_PCF") == 0)
        parameter.mode = Benchmark_Parameter::MODE_INSERT_ROTATE_PCF;
      else if (a.compare("INTERSECT_PCF") == 0)
        parameter.mode = Benchmark_Parameter::MODE_INTERSECT_PCF;
      else if (a.compare("LOAD_PCF") == 0)
        parameter.mode = Benchmark_Parameter::MODE_LOAD_PCF;
      else if (a.compare("MALLOC") == 0)
        parameter.mode = Benchmark_Parameter::MODE_MALLOC;
      else
        printf("Error parsing \"%s\"\n", a.c_str());
    }
    else if (a.compare("-r") == 0)
    {
      if (++i < argc)
        parameter.num_runs = atoi(argv[i]);
      else
        printf("Error parsing \"%s\"\n", argv[i]);
    }
    else if (a.compare("-f") == 0)
    {
      if (++i < argc)
        parameter.kinect_fps = atof(argv[i]);
      else
        printf("Error parsing \"%s\"\n", argv[i]);
    }
  }
}
}

namespace Provider {

char deleted_argument[] = "";

bool readIntValue(int argc, char **argv, int& i, std::string name, int& result)
{
  if (std::string(argv[i]).compare(name) == 0)
  {
    if (i + 1 < argc)
    {
      argv[i] = deleted_argument;
      result = atoi(argv[++i]);
      argv[i] = deleted_argument;
      return true;
    }
  }
  return false;
}

bool readFloatValue(int argc, char **argv, int& i, std::string name, float& result)
{
  if (std::string(argv[i]).compare(name) == 0)
  {
    if (i + 1 < argc)
    {
      argv[i] = deleted_argument;
      result = atof(argv[++i]);
      argv[i] = deleted_argument;
      return true;
    }
  }
  return false;
}

bool readBoolValue(int argc, char **argv, int& i, std::string name, bool& result)
{
  if (std::string(argv[i]).compare(name) == 0)
  {
    result = true;
    argv[i] = deleted_argument;
    return true;
  }
  return false;
}

bool parseArguments(vector<Provider_Parameter>& parameter, int argc, char **argv, bool report_error)
{
  bool error = false;
  char deleted_argument[] = "";

  string command = "";
  for (int i = 0; i < argc; ++i)
  {
    command += argv[i];
    command += " ";
  }

  for (int i = 1; i < argc; ++i)
  {
    std::string a = argv[i];

    if (a.length() != 0)
    {
      if (a.compare("-m") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          a = argv[++i];
          char* tmp = argv[i];
          argv[i] = deleted_argument;
          if (a.compare("load_pc") == 0)
            parameter.back().mode = Provider_Parameter::MODE_LOAD_PCF;
          else if (a.compare("kinect_live") == 0)
            parameter.back().mode = Provider_Parameter::MODE_KINECT_LIVE;
          else if (a.compare("ptu_live") == 0)
            parameter.back().mode = Provider_Parameter::MODE_PTU_LIVE;
          else if (a.compare("kinect_playback") == 0)
            parameter.back().mode = Provider_Parameter::MODE_KINECT_PLAYBACK;
          else if (a.compare("rand_plan") == 0)
            parameter.back().mode = Provider_Parameter::MODE_RANDOM_PLAN;
          else if (a.compare("ros") == 0)
            parameter.back().mode = Provider_Parameter::MODE_ROS;
          else if (a.compare("deserialize") == 0)
            parameter.back().mode = Provider_Parameter::MODE_DESERIALIZE;
          else
            argv[i] = tmp;
        }
      }
      else if (a.compare("-v") == 0)
      {
        argv[i] = deleted_argument;
        parameter.back().type = Provider_Parameter::TYPE_VOXEL_MAP;
      }
      else if (a.compare("-omap") == 0)
      {
        argv[i] = deleted_argument;
        parameter.back().type = Provider_Parameter::TYPE_OCTOMAP;
      }
      else if (a.compare("-mem") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().max_memory = size_t(atoi(argv[++i])) * gpu_voxels::cMBYTE2BYTE;
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-c") == 0)
      {
        parameter.back().collide = true;
        argv[i] = deleted_argument;
      }
      else if (a.compare("-f") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().pc_file = argv[++i];
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-id") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().kinect_id = argv[++i];
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-fps") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().kinect_fps = atof(argv[++i]);
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-resTree") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().resolution_tree = atoi(argv[++i]);
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-resOcc") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().resolution_occupied = atoi(argv[++i]);
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-resFree") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().resolution_free = atoi(argv[++i]);
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-shm") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.push_back(Provider_Parameter());
          parameter.back().shared_segment = atof(argv[++i]);
          parameter.back().command = command;
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-free") == 0)
      {
        argv[i] = deleted_argument;
        parameter.back().free_bounding_box = true;
      }
      else if (a.compare("-swap_x_z") == 0)
      {
        argv[i] = deleted_argument;
        parameter.back().swap_x_z = true;
      }
      else if (a.compare("-rebuild") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().rebuild_frame_count = atoi(argv[++i]);
          argv[i] = deleted_argument;
        }
      }
      else if (a.compare("-noFreeSpace") == 0)
      {
        argv[i] = deleted_argument;
        parameter.back().compute_free_space = false;
      }
      else if (a.compare("-maxRange") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          parameter.back().sensor_max_range = atoi(argv[++i]);
          argv[i] = deleted_argument;
        }
      }
      else if (readBoolValue(argc, argv, i, "-det", parameter.back().deterministic_octree))
      {
      }
      else if (Provider::readFloatValue(argc, argv, i, "-sx", parameter.back().plan_size.x))
      {
      }
      else if (Provider::readFloatValue(argc, argv, i, "-sy", parameter.back().plan_size.y))
      {
      }
      else if (Provider::readFloatValue(argc, argv, i, "-sz", parameter.back().plan_size.z))
      {
      }
      else if (Provider::readBoolValue(argc, argv, i, "-lb", parameter.back().voxelmap_intersect_with_lb))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-x", parameter.back().offset.x))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-y", parameter.back().offset.y))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-z", parameter.back().offset.z))
      {
      }
      else if (Provider::readBoolValue(argc, argv, i, "-serialize", parameter.back().serialize))
      {
      }
      else if (a.compare("-type") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = deleted_argument;
          a = argv[++i];
          char* tmp = argv[i];
          argv[i] = deleted_argument;
          if (a.compare("det") == 0)
            parameter.back().model_type = Provider_Parameter::eMT_Deterministc;
          else if (a.compare("prob") == 0)
            parameter.back().model_type = Provider_Parameter::eMT_Probabilistic;
          else if (a.compare("bit") == 0)
            parameter.back().model_type = Provider_Parameter::eMT_BitVector;
          else
            argv[i] = tmp;
        }
      }

//      else if (a.compare("-noFreeSpacePacking") == 0)
//      {
//        argv[i] = deleted_argument;
//        parameter.back().free_space_packing = false;
//      }
    }
  }

  if (report_error)
  {
    for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];
      if (a.length() != 0)
      {
        error = true;
        printf("Error parsing \"%s\"\n", a.c_str());
        break;
      }
    }
  }
  return error;
}

bool readPcFile(vector<Provider_Parameter>& parameter)
{
  for (uint32_t i = 0; i < parameter.size(); ++i)
  {
    if (parameter[i].pc_file.empty()
        && (parameter[i].mode == Provider_Parameter::MODE_LOAD_PCF
            || parameter[i].mode == Provider_Parameter::MODE_RANDOM_PLAN
            || parameter[i].mode == Provider_Parameter::MODE_KINECT_PLAYBACK))
    {
      printf("File name missing!\n");
      return false;
    }
    else if(!parameter[i].pc_file.empty() && parameter[i].mode != Provider_Parameter::MODE_KINECT_PLAYBACK
        && parameter[i].mode != Provider_Parameter::MODE_DESERIALIZE)
    {

      bool res = file_handling::PointcloudFileHandler::Instance()->loadPointCloud(parameter[i].pc_file, true, parameter[i].points);
      if(res)
      {
        parameter[i].num_points = parameter[i].points.size();
        if (parameter[i].swap_x_z)
        {
          for (size_t j = 0; j < parameter[i].num_points; ++j)
            parameter[i].points[j] = Vector3f(parameter[i].points[j].z, parameter[i].points[j].y, parameter[i].points[j].x);
        }

        return true;
      }
    }
  }
  return false;
}
}

namespace Bench {

bool parseArguments(Bech_Parameter& parameter, int argc, char **argv, bool report_error)
{
  string command = "";
  for (int i = 0; i < argc; ++i)
  {
    command += argv[i];
    command += " ";
  }
  parameter.command = command;

  bool error = parseArguments(parameter.provider_parameter, argc, argv, false);

  // parse my arguments

  for (int i = 1; i < argc; ++i)
  {
    std::string a = argv[i];

    if (a.length() != 0)
    {
      if (a.compare("-b") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = Provider::deleted_argument;
          a = argv[++i];
          char* tmp = argv[i];
          argv[i] = Provider::deleted_argument;
          if (a.compare("build") == 0)
            parameter.mode = Bech_Parameter::MODE_BUILD;
          else if (a.compare("insert") == 0)
            parameter.mode = Bech_Parameter::MODE_INSERT;
          else if (a.compare("collide_live") == 0)
            parameter.mode = Bech_Parameter::MODE_COLLIDE_LIVE;
          else if (a.compare("collide") == 0)
            parameter.mode = Bech_Parameter::MODE_COLLIDE;
          else
            argv[i] = tmp;
        }
      }
      else if (a.compare("-runs") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = Provider::deleted_argument;
          parameter.runs = atoi(argv[++i]);
          argv[i] = Provider::deleted_argument;
        }
      }
      else if (a.compare("-resFrom") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = Provider::deleted_argument;
          parameter.resolution_from = atoi(argv[++i]);
          argv[i] = Provider::deleted_argument;
        }
      }
      else if (a.compare("-resTo") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = Provider::deleted_argument;
          parameter.resolution_to = atoi(argv[++i]);
          argv[i] = Provider::deleted_argument;
        }
      }
      else if (a.compare("-resStep") == 0)
      {
        if (i + 1 < argc)
        {
          argv[i] = Provider::deleted_argument;
          parameter.resolution_scaling = atof(argv[++i]);
          argv[i] = Provider::deleted_argument;
        }
      }
      else if (Provider::readIntValue(argc, argv, i, "-blocksFrom", parameter.blocks_from))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-blocksTo", parameter.blocks_to))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-blocksStep", parameter.blocks_step))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-threadsFrom", parameter.threads_from))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-threadsTo", parameter.threads_to))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-threadsStep", parameter.threads_step))
      {
      }
      else if (Provider::readBoolValue(argc, argv, i, "-logRuns", parameter.log_runs))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-replay", parameter.replay))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-clevelFrom", parameter.collision_level_from))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-clevelTo", parameter.collision_level_to))
      {
      }
      else if (Provider::readIntValue(argc, argv, i, "-clevelStep", parameter.collision_level_step))
      {
      }
      else if (Provider::readBoolValue(argc, argv, i, "-saveCollisions", parameter.save_collisions))
      {
      }
    }
  }

  if (report_error)
  {
    for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];
      if (a.length() != 0)
      {
        error = true;
        printf("Error parsing \"%s\"\n", a.c_str());
        break;
      }
    }
  }
  return error;
}

}


}
}

