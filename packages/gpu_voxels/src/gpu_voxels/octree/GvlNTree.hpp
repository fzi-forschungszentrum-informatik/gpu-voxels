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
 * \date    2014-07-07
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_GVL_NTREE_HPP_INCLUDED
#define GPU_VOXELS_OCTREE_GVL_NTREE_HPP_INCLUDED

#include <gpu_voxels/octree/GvlNTree.h>
#include <gpu_voxels/octree/Octree.h>
#include <gpu_voxels/voxel/BitVoxel.hpp>

namespace gpu_voxels {
namespace NTree {

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::GvlNTree(const float voxel_side_length, const MapType map_type) :
    base(NUM_BLOCKS, NUM_THREADS_PER_BLOCK, uint32_t(voxel_side_length * 1000))
{
  this->m_map_type = map_type;

  m_d_free_space_voxel2 = NULL;
  m_d_object_voxel2 = NULL;

  // setup sensor parameter
  m_sensor.object_data.m_initial_probability = INITIAL_OCCUPIED_PROBABILITY;
  m_sensor.object_data.m_update_probability = OCCUPIED_UPDATE_PROBABILITY;
  m_sensor.object_data.m_invalid_measure = 0;
  m_sensor.object_data.m_remove_max_range_data = true;
  m_sensor.object_data.m_sensor_range = 7.0;
  m_sensor.object_data.m_use_invalid_measures = false;
  m_sensor.object_data.m_process_data = true;

  m_sensor.free_space_data = m_sensor.object_data; // copy data which doesn't matter

  // probabilities for free space aren't used for preprocessing of sensor data
  m_sensor.free_space_data.m_cut_x_boarder = KINECT_CUT_FREE_SPACE_X;
  m_sensor.free_space_data.m_cut_y_boarder = KINECT_CUT_FREE_SPACE_Y;
  m_sensor.free_space_data.m_invalid_measure = 0;
  m_sensor.free_space_data.m_remove_max_range_data = false;
  m_sensor.free_space_data.m_sensor_range = 7.0;
  m_sensor.free_space_data.m_use_invalid_measures = true;
  m_sensor.free_space_data.m_process_data = true; // parameter.compute_free_space;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::~GvlNTree()
{

}

// ------ BEGIN Global API functions ------
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertPointCloud(
    const std::vector<Vector3f> &point_cloud, BitVoxelMeaning voxelType)
{
  lock_guard guard(this->m_mutex);
  if (voxelType != eBVM_OCCUPIED)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
  {
    // Copy points to gpu and tranform to voxel coordinates
    thrust::host_vector<Vector3f> h_points(point_cloud.begin(), point_cloud.end());
    thrust::device_vector<Vector3ui> d_voxels;
    this->toVoxelCoordinates(h_points, d_voxels);

    insertVoxelData(d_voxels);
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertPointCloud(const PointCloud &pointcloud, const BitVoxelMeaning voxel_meaning)
{
  lock_guard guard(this->m_mutex);

  if (voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
  {
    thrust::device_vector<Vector3ui> d_voxels(pointcloud.getPointCloudSize());

    kernel_toVoxels<<<this->numBlocks, this->numThreadsPerBlock>>>(pointcloud.getConstDevicePointer(), pointcloud.getPointCloudSize(), D_PTR(d_voxels), float(this->m_resolution / 1000.0f));
    CHECK_CUDA_ERROR();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    insertVoxelData(d_voxels);
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertCoordinateList(const std::vector<Vector3ui> &coordinates, const BitVoxelMeaning voxel_meaning)
{
  lock_guard guard(this->m_mutex);
  if (voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
  {
    // Copy points to gpu and tranform to voxel coordinates
    thrust::host_vector<Vector3ui> h_coordinates(coordinates.begin(), coordinates.end());
    thrust::device_vector<Vector3ui> d_voxels;
    d_voxels = h_coordinates;    

    insertVoxelData(d_voxels);
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertCoordinateList(const Vector3ui *d_coordinates, uint32_t size, const BitVoxelMeaning voxel_meaning)
{
  lock_guard guard(this->m_mutex);
  if (voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
  {
    // Copy points to gpu and tranform to voxel coordinates
    thrust::device_vector<Vector3ui> d_voxels(d_coordinates, d_coordinates + size);

    insertVoxelData(d_voxels);
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertPointCloudWithFreespaceCalculation(
    const std::vector<Vector3f> &point_cloud_in_sensor_coords, const Matrix4f &sensor_pose,
    uint32_t free_space_resolution, uint32_t occupied_space_resolution)
{
  lock_guard guard(this->m_mutex);
  m_sensor.free_space_data.m_voxel_side_length = free_space_resolution;
  m_sensor.object_data.m_voxel_side_length = occupied_space_resolution;

  m_sensor.pose = sensor_pose;

  m_sensor.data_width = point_cloud_in_sensor_coords.size();
  m_sensor.data_height = 1;

  // processSensorData() will allcate space for d_free_space_voxel and d_object_voxel if they are NULL
  m_sensor.processSensorData(point_cloud_in_sensor_coords.data(), m_d_free_space_voxel2, m_d_object_voxel2);
  // convert sensor origin in discrete coordinates of the NTree
  gpu_voxels::Vector3ui sensor_origin = gpu_voxels::Vector3ui(
      uint32_t(sensor_pose.a14 * 1000.0f / this->m_resolution),
      uint32_t(sensor_pose.a24 * 1000.0f / this->m_resolution),
      uint32_t(sensor_pose.a34 * 1000.0f / this->m_resolution));

  this->insertVoxel(*m_d_free_space_voxel2, *m_d_object_voxel2, sensor_origin,
                    free_space_resolution, occupied_space_resolution);

  //CAUTION: Check for needs_rebuild after inserting new data!
}

//Collision Interface Implementation
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
    const voxelmap::BitVectorVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  return collideWithResolution(map, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
    const voxelmap::ProbVoxelMap *map, float coll_threshold, const Vector3i &offset)
{
  return collideWithResolution(map, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
    const voxellist::BitVectorVoxelList *map, float coll_threshold, const Vector3i &offset)
{
  return collideWithResolution(map, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
    const GvlNTreeDet *map, float coll_threshold, const Vector3i &offset)
{
  return collideWithResolution(map, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
    const GvlNTreeProb *map, float coll_threshold, const Vector3i &offset)
{
  return collideWithResolution(map, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
    const voxellist::BitVectorMortonVoxelList *map, float coll_threshold, const Vector3i &offset)
{
  return collideWithResolution(map, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const voxelmap::BitVectorVoxelMap *map, float coll_threshold, const uint32_t resolution_level, const Vector3i &offset)
{
  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  //voxelmap::BitVectorVoxelMap* _voxelmap = dynamic_cast<voxelmap::BitVectorVoxelMap*>(map);
  voxelmap::BitVectorVoxelMap* _voxelmap = (voxelmap::BitVectorVoxelMap*)map;
  if (_voxelmap == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelMap' failed!" << endl);
  size_t temp = this->template intersect_sparse<true, false, false, BitVectorVoxel>(*_voxelmap, NULL, 0, offset, NULL);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const voxelmap::ProbVoxelMap *map, float coll_threshold, const uint32_t resolution_level, const Vector3i &offset)
{
  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  //voxelmap::ProbVoxelMap* _voxelmap = dynamic_cast<voxelmap::ProbVoxelMap*>(map);
  voxelmap::ProbVoxelMap* _voxelmap = (voxelmap::ProbVoxelMap*)map;
  if (_voxelmap == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'VoxelList' failed!" << endl);
  size_t temp = this->template intersect_sparse<true, false, false, ProbabilisticVoxel>(
      *_voxelmap, NULL, 0, offset, NULL);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const voxellist::BitVectorVoxelList *map, float coll_threshold, const uint32_t resolution_level, const Vector3i &offset)
{
  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  //voxellist::BitVectorVoxelList* _voxellist = dynamic_cast<voxellist::BitVectorVoxelList*>(map);
  voxellist::BitVectorVoxelList* _voxellist = (voxellist::BitVectorVoxelList*)map;
  if (_voxellist == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelList' failed!" << endl);
  size_t temp = this->template intersect_sparse<true, false, false, BitVectorVoxel>(*_voxellist, NULL, 0, offset, NULL);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const GvlNTreeDet *map, float coll_threshold, const uint32_t resolution_level, const Vector3i &offset)
{
  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  const bool save_collisions = true;
  //NTreeDet* _ntree = dynamic_cast<NTreeDet*>(map);
  NTreeDet* _ntree = (NTreeDet*)map;
  if (_ntree == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "DYNAMIC CAST TO 'NTreeDet' FAILED!" << endl);
  if(offset != Vector3i())
    LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);
  size_t temp = this->template intersect_load_balance<>(_ntree, resolution_level, DefaultCollider(), save_collisions);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const GvlNTreeProb *map, float coll_threshold, const uint32_t resolution_level, const Vector3i &offset)
{
  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  const bool save_collisions = true;
  //NTreeProb* _ntree = dynamic_cast<NTreeProb*>(map);
  NTreeProb* _ntree = (NTreeProb*)map;
  if (_ntree == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "DYNAMIC CAST TO 'NTreeProb' FAILED!" << endl);
  if(offset != Vector3i())
    LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);
  size_t temp = this->template intersect_load_balance<>(_ntree, resolution_level, DefaultCollider(), save_collisions);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const voxellist::BitVectorMortonVoxelList *map, float coll_threshold, const uint32_t resolution_level, const Vector3i &offset)
{
  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return SSIZE_MAX;
  }
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  //voxellist::BitVectorMortonVoxelList* _voxellist = dynamic_cast<voxellist::BitVectorMortonVoxelList*>(map);
  voxellist::BitVectorMortonVoxelList* _voxellist = (voxellist::BitVectorMortonVoxelList*)map;
  if (_voxellist == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorMortonVoxelList' failed!" << endl);
  if(offset != Vector3i())
    LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);

  // This previously used "intersect<BIT_VECTOR_LENGTH, true, false>" which itself utilized a more effective kernel.
  size_t temp = this->template intersect_morton<true, false, false, BitVectorVoxel>(*_voxellist);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypes(
    const voxelmap::BitVectorVoxelMap *map, BitVectorVoxel &types_in_collision, float coll_threshold, const Vector3i &offset)
{
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  //voxelmap::BitVectorVoxelMap* _voxelmap = dynamic_cast<voxelmap::BitVectorVoxelMap*>(map);
  voxelmap::BitVectorVoxelMap* _voxelmap = (voxelmap::BitVectorVoxelMap*)map;
  if (_voxelmap == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelMap' failed!" << endl);
  size_t temp = this->template intersect_sparse<true, true, false, BitVectorVoxel>(*_voxelmap, &types_in_collision, 0, offset, NULL);
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypes(
    const voxellist::BitVectorVoxelList *map, BitVectorVoxel &types_in_collision, float coll_threshold, const Vector3i &offset)
{
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  //voxellist::BitVectorVoxelList* _voxellist = dynamic_cast<voxellist::BitVectorVoxelList*>(map);
  voxellist::BitVectorVoxelList* _voxellist = (voxellist::BitVectorVoxelList*)map;
  if (_voxellist == NULL)
    LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelList' failed!" << endl);
  size_t temp = this->template intersect_sparse<true, true, false, BitVectorVoxel>(*_voxellist, &types_in_collision, 0, offset, NULL);
  return temp;
}


template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypesConsideringUnknownCells(
    const GpuVoxelsMapSharedPtr map, BitVectorVoxel& types_in_collision, size_t &num_colls_with_unknown_cells, const Vector3i &offset)
{
  boost::lock(this->m_mutex, map->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(map->m_mutex, boost::adopt_lock);

  size_t num_collisions = SSIZE_MAX;
  num_colls_with_unknown_cells = SSIZE_MAX;
  MapType type = map->getMapType();
  voxel_count tmp_cols_w_unknown;

  if (type == MT_BITVECTOR_VOXELMAP)
  {
    voxelmap::BitVectorVoxelMap* _voxelmap = dynamic_cast<voxelmap::BitVectorVoxelMap*>(map.get());
    if (_voxelmap == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelMap' failed!" << endl);
    num_collisions = this->template intersect_sparse<true, true, true, BitVectorVoxel>(*_voxelmap, &types_in_collision, 0, offset, &tmp_cols_w_unknown);
    num_colls_with_unknown_cells = tmp_cols_w_unknown;
  }
  else if (type == MT_BITVECTOR_VOXELLIST)
  {
    voxellist::BitVectorVoxelList* _voxellist = dynamic_cast<voxellist::BitVectorVoxelList*>(map.get());
    if (_voxellist == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelList' failed!" << endl);

    num_collisions = this->template intersect_sparse<true, true, true, BitVectorVoxel>(*_voxellist, &types_in_collision, 0, offset, &tmp_cols_w_unknown);
    num_colls_with_unknown_cells = tmp_cols_w_unknown;
  }
  else
    LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);

  return num_collisions;
}


template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithBitcheck(
    const GpuVoxelsMapSharedPtr map, const u_int8_t margin, const Vector3i &offset)
{
  size_t num_collisions = SSIZE_MAX;

  switch (map->getMapType())
  {
    case MT_BITVECTOR_VOXELMAP:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      break;
    }
    case MT_BITVECTOR_OCTREE:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      break;
    }
    default:
    {
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
      break;
    }
  }
  return num_collisions;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloudWithSelfCollisionCheck(
                                                        const MetaPointCloud *robot_links,
                                                        const std::vector<BitVoxelMeaning>& voxel_meanings,
                                                        const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks,
                                                        BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
{
  LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return true;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloud(
    const MetaPointCloud &meta_point_cloud, BitVoxelMeaning voxelType)
{
  lock_guard guard(this->m_mutex);
  if (voxelType != eBVM_OCCUPIED)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);

  // Get adress from device
  Vector3f* d_points = NULL;
  MetaPointCloudStruct tmp_struct;
  HANDLE_CUDA_ERROR(cudaMemcpy(&tmp_struct, meta_point_cloud.getDeviceConstPointer(), sizeof(MetaPointCloudStruct), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(&d_points, tmp_struct.clouds_base_addresses, sizeof(Vector3f*), cudaMemcpyDeviceToHost));

  size_t num_points = meta_point_cloud.getAccumulatedPointcloudSize();
  thrust::device_vector<Vector3ui> d_voxels(num_points);
  kernel_toVoxels<<<this->numBlocks, this->numThreadsPerBlock>>>(d_points, num_points, D_PTR(d_voxels), this->m_resolution / 1000.0f);
  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  insertVoxelData(d_voxels);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloud(
    const MetaPointCloud& meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings)
{
  /* Basically this is a dummy implementation since the method can't be left abstract.
     However, I'm not sure whether this functionality makes sense here, so I didn't
     implement it.
   */
  LOGGING_WARNING_C(OctreeLog, NTree, "This functionality is not implemented, yet. The pointcloud will be inserted with the first BitVoxelMeaning." << endl);
  insertMetaPointCloud(meta_point_cloud, voxel_meanings.front());
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::merge(
    const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset, const BitVoxelMeaning* new_meaning)
{
  LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return false;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::merge(
    const GpuVoxelsMapSharedPtr other, const Vector3i &voxel_offset, const BitVoxelMeaning* new_meaning)
{
  LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return false;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
std::size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getMemoryUsage() const
{
  lock_guard guard(this->m_mutex);
  std::size_t temp = this->getMemUsage();
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::clearMap()
{
  lock_guard guard(this->m_mutex);
  this->clear();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  if (voxel_meaning != eBVM_OCCUPIED)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
  else
    clearMap();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::writeToDisk(const std::string path)
{
  lock_guard guard(this->m_mutex);
  std::ofstream out(path.c_str());
  if(!out.is_open())
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "Write to file " << path << " failed!" <<  endl);
    return false;
  }
  this->serialize(out);
  out.close();
  return true;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::readFromDisk(const std::string path)
{
  lock_guard guard(this->m_mutex);
  std::ifstream in(path.c_str());
  if(!in.is_open())
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "Read from file " << path << " failed!" << endl);
    return false;
  }
  this->deserialize(in);
  in.close();
  return true;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild() const
{
  lock_guard guard(this->m_mutex);
  bool temp = this->NTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild();
  return temp;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::rebuild()
{
  lock_guard guard(this->m_mutex);
  this->NTree<branching_factor, level_count, InnerNode, LeafNode>::rebuild();
  return true;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
Vector3ui GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getDimensions() const
{
  uint32_t s = static_cast<uint32_t>(getVoxelSideLength<branching_factor>(level_count - 1));
  return Vector3ui(s, s, s);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
Vector3f GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getMetricDimensions() const
{
  Vector3ui dim_in_voxel = getDimensions();
  return Vector3f(dim_in_voxel.x, dim_in_voxel.z, dim_in_voxel.z) * float(base::m_resolution / 1000.0f);
}

// ------ END Global API functions ------

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertVoxelData(
    thrust::device_vector<Vector3ui> &d_voxels)
{
  uint32_t num_points = d_voxels.size();
  if(num_points > 0)
  {
    if (this->m_has_data)
    {
      // Have to insert voxels and adjust occupancy since there are already some voxels in the NTree
      // Transform voxel coordinates to morton code
      thrust::device_vector<OctreeVoxelID> d_voxels_morton(num_points);
      kernel_toMortonCode<<<this->numBlocks, this->numThreadsPerBlock>>>(D_PTR(d_voxels), num_points,
      D_PTR(d_voxels_morton));
      CHECK_CUDA_ERROR();
      HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

      // Sort and remove duplicates
      // TODO: remove thrust::unique() and adapt NTree::insert() to handle duplicates in the input data
      thrust::sort(d_voxels_morton.begin(), d_voxels_morton.end());
      thrust::device_vector<OctreeVoxelID>::iterator new_end = thrust::unique(d_voxels_morton.begin(),
                                                                        d_voxels_morton.end());
      size_t num_voxel_unique = new_end - d_voxels_morton.begin();

      // Insert voxels
      typename base::BasicData tmp;
      getHardInsertResetData(tmp);
      thrust::constant_iterator<typename base::BasicData> reset_data(tmp);
      getOccupiedData(tmp);
      thrust::constant_iterator<typename base::BasicData> set_basic_data(tmp);
      this->template insertVoxel<true, typename base::BasicData>(D_PTR(d_voxels_morton), set_basic_data,reset_data, num_voxel_unique, 0);

      // Recover tree invariant
      this->propagate(uint32_t(num_voxel_unique));
    }
    else
    {
      // Use plain this->build since NTree is empty
      this->build(d_voxels, false);
    }
  }
}

}  // end of ns
}  // end of ns

#endif
