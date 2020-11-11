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
 * \date    2014-07-08
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_PROB_VOXELMAP_H_INCLUDED

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxel/ProbabilisticVoxel.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/CollisionInterfaces.h>
#include <sensor_msgs/PointCloud2.h>

namespace gpu_voxels {
namespace voxelmap {

class ProbVoxelMap: public TemplateVoxelMap<ProbabilisticVoxel>,
    public CollidableWithBitVectorVoxelMap, public CollidableWithProbVoxelMap
{
public:
  typedef ProbabilisticVoxel Voxel;
  typedef TemplateVoxelMap<Voxel> Base;

  ProbVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type);
  ProbVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type);
  virtual ~ProbVoxelMap();

  template<std::size_t length>
  void insertSensorData(const PointCloud &global_points, const Vector3f &sensor_pose, const bool enable_raycasting, const bool cut_real_robot,
                        const BitVoxelMeaning robot_voxel_meaning, const Probability prob, BitVoxel<length>* robot_map = NULL);

  virtual bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud *robot_links,
                                                          const std::vector<BitVoxelMeaning>& voxel_meanings = std::vector<BitVoxelMeaning>(),
                                                          const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks = std::vector<BitVector<BIT_VECTOR_LENGTH> >(),
                                                          BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = NULL);

  virtual void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning);

  virtual MapType getTemplateType() const { return MT_PROBAB_VOXELMAP; }

  // Collision Interface Methods

  size_t collideWith(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());
  size_t collideWith(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, const Vector3i &offset = Vector3i());

  virtual void moveInto(ProbVoxelMap& dest, const Vector3f offset) const;
  virtual void move(Voxel* dest_data, const Voxel* src_data, const Vector3f offset) const;
  virtual void publishPointcloud(sensor_msgs::PointCloud2* pointcloud_msg, const float occupancyThreshold=0.5);
};

} // end of namespace
} // end of namespace

#endif
