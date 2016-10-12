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
 * \date    2016-11-02
 *
 */
//----------------------------------------------------------------------

#include <iostream>
#include <boost/filesystem/path.hpp>
#include "HeightMapLoader.h"
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace gpu_voxels {
namespace file_handling {

void HeightMapLoader(std::string bottom_map, std::string ceiling_map, bool use_model_path,
                                 size_t bottom_start_height, size_t ceiling_end_height,
                                 float meter_per_pixel, float meter_per_greyshade,
                                 gpu_voxels::Vector3f metric_offset, PointCloud &cloud)
{
  std::string bottom_map_path;
  std::string ceiling_map_path;

  bottom_map_path = (getGpuVoxelsPath(use_model_path) / boost::filesystem::path(bottom_map)).string();
  ceiling_map_path = (getGpuVoxelsPath(use_model_path) / boost::filesystem::path(ceiling_map)).string();

  unsigned char *simulatedEnvBottom;
  unsigned char *simulatedEnvCeiling;
  int MapDimX, MapDimY;
  int comp;

  if(!bottom_map.empty())
  {
    simulatedEnvBottom = stbi_load(bottom_map_path.c_str(), &MapDimX, &MapDimY, &comp, STBI_default);
    if(!simulatedEnvBottom)
    {
      LOGGING_ERROR_C(Gpu_voxels_helpers, HeightMapLoader, "Could not read bottom map from " << bottom_map_path << endl);
    }else{
      LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Read bottom map from " << bottom_map_path << " with size: " << MapDimX << "x" << MapDimY << " and " << comp << " components." << endl);
      if(comp != 1)
      {
        LOGGING_ERROR_C(Gpu_voxels_helpers, HeightMapLoader, "Bottom map not given in Greyscale and without Alpha. Not using the map!" << endl);
        simulatedEnvBottom = NULL;
      }
    }
  }else{
    simulatedEnvBottom = NULL;
    LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Not using a bottom map." << endl);
  }

  if(!ceiling_map.empty())
  {
    int cMapDimX, cMapDimY;
    simulatedEnvCeiling = stbi_load(ceiling_map_path.c_str(), &cMapDimX, &cMapDimY, &comp, STBI_default);
    if(!simulatedEnvCeiling)
    {
      LOGGING_ERROR_C(Gpu_voxels_helpers, HeightMapLoader, "Could not read ceiling map from " << ceiling_map_path << endl);
    }else{
      LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Read ceiling map from " << ceiling_map_path << " with size: " << cMapDimX << "x" << cMapDimY << " and " << comp << " components." << endl);
      if(cMapDimX != MapDimX || cMapDimY != MapDimY)
      {
        LOGGING_ERROR_C(Gpu_voxels_helpers, HeightMapLoader, "Ceiling map dimension differs from floor map! Not loading!" << endl);
        simulatedEnvCeiling = NULL;
      }
      if(comp != 1)
      {
        LOGGING_ERROR_C(Gpu_voxels_helpers, HeightMapLoader, "Ceiling map not given in Greyscale and without Alpha. Not using the map!" << endl);
        simulatedEnvCeiling = NULL;
      }
    }
  }else{
    simulatedEnvCeiling = NULL;
    LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Not using a ceiling map." << endl);
  }


  std::vector<Vector3f> points;
  bool from_bottom_flag;
  bool from_ceiling_flag;
  unsigned char bottom_value;
  unsigned char ceiling_value;

  for(size_t height_level = bottom_start_height; height_level <= ceiling_end_height; height_level++)
  {
    for (size_t coordX = 0; coordX < size_t(MapDimX); ++coordX)
    {
      for (size_t coordY = 0; coordY < size_t(MapDimY); ++coordY)
      {
        if(simulatedEnvBottom)
        {
          bottom_value = simulatedEnvBottom[coordY*MapDimX + coordX];
          from_bottom_flag = (int(bottom_value) >= height_level);
          //LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Bottom height "<< height_level <<" Pixel at [" << coordX << ", " << coordY << "] has value " << int(bottom_value) << endl);
        }else{
          bottom_value = 0;
          from_bottom_flag = false;
        }

        if(simulatedEnvCeiling)
        {
          ceiling_value = simulatedEnvCeiling[coordY*MapDimX + coordX];
          from_ceiling_flag = (int(ceiling_value) <= height_level);
          //LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Ceiling height "<< height_level <<" Pixel at [" << coordX << ", " << coordY << "] has value " << int(ceiling_value) << endl);
        }else{
          ceiling_value = 255;
          from_ceiling_flag = false;
        }

        if (from_bottom_flag || from_ceiling_flag)
        {
          Vector3f point;
          point.x = float(0.5 + coordX) * meter_per_pixel;  // scale from pixel to m
          point.y = float(0.5 + MapDimY - coordY) * meter_per_pixel; // scale from pixel to m
          point.z = float(0.5 + height_level) * meter_per_greyshade; // scale from intensity to m
          points.push_back(point + metric_offset);
        }
      }
    }
  }
  LOGGING_INFO_C(Gpu_voxels_helpers, HeightMapLoader, "Created a heightmap with " << points.size() << " points." << endl);
  cloud.update(points);
}



}  // end of namespace
}  // end of namespace
