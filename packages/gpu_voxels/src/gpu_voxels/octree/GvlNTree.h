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
 * \date    2014-07-04
 *
 */
//----------------------------------------------------------------------/*
#ifndef  GPU_VOXELS_OCTREE_GVL_NTREE_H_INCLUDED
#define  GPU_VOXELS_OCTREE_GVL_NTREE_H_INCLUDED

#include <gpu_voxels/GpuVoxelsMap.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/voxelmap/BitVoxel.h>

namespace gpu_voxels {
namespace NTree {

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
class GvlNTree : public GpuVoxelsMap, public NTree<branching_factor, level_count, InnerNode, LeafNode>
{
public:
  typedef NTree<branching_factor, level_count, InnerNode, LeafNode> base;

  GvlNTree(const float voxel_side_length, const MapType map_type);
  virtual ~GvlNTree();

  // ------ START Global API functions ------

  virtual void insertGlobalData(const std::vector<Vector3f> &point_cloud, VoxelType voxel_type);

  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, VoxelType voxel_type);

  virtual void clearMap();

  virtual void clearVoxelType(VoxelType voxel_type);

  virtual size_t collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0);

  virtual size_t collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const uint32_t resolution_level = 0);

  virtual size_t collideWithTypes(const GpuVoxelsMapSharedPtr other, voxelmap::BitVectorVoxel&  types_in_collision, float coll_threshold = 1.0);

  virtual bool insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test);

  virtual std::size_t getMemoryUsage();

  virtual void writeToDisk(const std::string path);

  virtual bool readFromDisk(const std::string path);

  virtual bool needsRebuild();

  virtual bool rebuild();

  // ------ END Global API functions ------

protected:
  virtual void insertVoxelData(thrust::device_vector<Vector3ui> &d_voxels);
};

}  // end of ns
}  // end of ns


#endif /* GPUVOXELSNTREE_H_ */
