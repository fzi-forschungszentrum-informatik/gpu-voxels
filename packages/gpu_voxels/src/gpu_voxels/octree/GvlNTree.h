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
#include <gpu_voxels/voxel/BitVoxel.h>

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

  virtual void insertPointCloud(const std::vector<Vector3f> &point_cloud, BitVoxelMeaning voxel_meaning);

  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, BitVoxelMeaning voxel_meaning);

  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings);

  virtual void clearMap();

  virtual void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning);

  virtual size_t collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const Vector3ui &offset = Vector3ui());

  virtual size_t collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3ui &offset = Vector3ui());

  virtual size_t collideWithTypes(const GpuVoxelsMapSharedPtr other, BitVectorVoxel&  types_in_collision, float coll_threshold = 1.0, const Vector3ui &offset = Vector3ui());

  virtual size_t collideWithBitcheck(const GpuVoxelsMapSharedPtr other, const u_int8_t margin = 0, const Vector3ui &offset = Vector3ui());

  virtual bool insertRobotConfiguration(const MetaPointCloud *robot_links, bool with_self_collision_test);

  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset = Vector3f(), const BitVoxelMeaning* new_meaning = NULL);
  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3ui &voxel_offset = Vector3ui(), const BitVoxelMeaning* new_meaning = NULL);

  virtual std::size_t getMemoryUsage() const;

  virtual bool writeToDisk(const std::string path);

  virtual bool readFromDisk(const std::string path);

  virtual bool needsRebuild() const;

  virtual bool rebuild();

  virtual Vector3ui getDimensions() const;

  virtual Vector3f getMetricDimensions() const;

  // ------ END Global API functions ------

protected:
  virtual void insertVoxelData(thrust::device_vector<Vector3ui> &d_voxels);
};

}  // end of ns
}  // end of ns


#endif /* GPUVOXELSNTREE_H_ */
