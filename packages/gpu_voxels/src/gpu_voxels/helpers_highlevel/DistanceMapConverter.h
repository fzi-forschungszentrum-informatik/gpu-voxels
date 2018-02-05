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
 * \date    2017-04-09
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/voxellist/BitVoxelList.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>

namespace gpu_voxels
{
namespace distance_map_converter
{

struct in_range
{
  free_space_t min_dist;
  free_space_t max_dist;

  __host__ __device__
  in_range(free_space_t min_dist_, free_space_t max_dist_) :
    min_dist(min_dist_), max_dist(max_dist_)
  {
    printf("Contructed in_range operator \n");
  }

  __host__ __device__
  bool operator()(const free_space_t &dist) const
  {
    return (dist >= min_dist && dist <= max_dist);
  }
};

struct in_range_tuple
{
  free_space_t min_dist;
  free_space_t max_dist;

  __host__ __device__
  in_range_tuple(free_space_t min_dist_, free_space_t max_dist_) :
    min_dist(min_dist_), max_dist(max_dist_) {}

  __host__ __device__
  bool operator()(const thrust::tuple<free_space_t, MapVoxelID> &dist_tuple) const
  {
    return (thrust::get<0>(dist_tuple) >= min_dist && thrust::get<0>(dist_tuple) <= max_dist);
  }
};

size_t extract_given_distances(const voxelmap::DistanceVoxelMap& dist_map,
                               free_space_t min_sqare_dist, free_space_t max_sqare_dist,
                               voxellist::BitVectorVoxelList& result);

}
}
