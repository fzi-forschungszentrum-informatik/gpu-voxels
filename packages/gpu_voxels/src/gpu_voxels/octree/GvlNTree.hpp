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
#include <gpu_voxels/voxelmap/BitVoxel.hpp>

namespace gpu_voxels {
namespace NTree {

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::GvlNTree(const float voxel_side_length, const MapType map_type) :
    base(NUM_BLOCKS, NUM_THREADS_PER_BLOCK, uint32_t(voxel_side_length * 1000))
{
  this->m_map_type = map_type;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::~GvlNTree()
{

}

// ------ BEGIN Global API functions ------
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertGlobalData(
    const std::vector<Vector3f> &point_cloud, VoxelType voxelType)
{
  if (voxelType != 0)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_VT_0 << endl);
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
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold, const Vector3ui &offset)
{
  return collideWithResolution(other, coll_threshold, 0, offset);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
    const GpuVoxelsMapSharedPtr other, float coll_threshold, const uint32_t resolution_level, const Vector3ui &offset)
{
  size_t num_collisions = SSIZE_MAX;

  if(resolution_level >= level_count)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" <<  endl);
    return num_collisions;
  }

  GpuVoxelsMap* map = other.get();
  MapType type = map->getMapType();

  const bool save_collisions = true;
  const uint32_t min_level = resolution_level;

  if (type == MT_BITVECTOR_OCTREE)
  {
    NTreeDet* _ntree = dynamic_cast<NTreeDet*>(map);
    if (_ntree == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "DYNAMIC CAST TO 'NTreeDet' FAILED!" << endl);
    if(offset != Vector3ui())
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);
    num_collisions = this->template intersect_load_balance<>(_ntree, min_level, DefaultCollider(), save_collisions);
  }
  else if (type == MT_PROBAB_OCTREE)
  {
    NTreeProb* _ntree = dynamic_cast<NTreeProb*>(map);
    if (_ntree == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "DYNAMIC CAST TO 'NTreeProb' FAILED!" << endl);
    if(offset != Vector3ui())
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);
    num_collisions = this->template intersect_load_balance<>(_ntree, min_level, DefaultCollider(), save_collisions);
  }
  else if (type == MT_PROBAB_VOXELMAP)
  {
    voxelmap::ProbVoxelMap* _voxelmap = dynamic_cast<voxelmap::ProbVoxelMap*>(map);
    if (_voxelmap == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'VoxelList' failed!" << endl);

    num_collisions = this->template intersect_sparse<true, false, voxelmap::ProbabilisticVoxel>(
        *_voxelmap, NULL, 0, offset);
//    num_collisions = this->template intersect_load_balance<VOXELMAP_FLAG_SIZE, true, false, gpu_voxels::Voxel, true>(
//        *_voxelmap);
  }
  else if (type == MT_BITVECTOR_VOXELMAP)
  {
    voxelmap::BitVectorVoxelMap* _voxelmap = dynamic_cast<voxelmap::BitVectorVoxelMap*>(map);
    if (_voxelmap == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelMap' failed!" << endl);
    num_collisions = this->template intersect_sparse<true, false, voxelmap::BitVectorVoxel>(*_voxelmap, NULL, 0, offset);
    //    num_collisions = this->template intersect_load_balance<VOXELMAP_FLAG_SIZE, true, false, gpu_voxels::Voxel, true>(
    //        *_voxelmap);
  }
  else if (type == MT_BITVECTOR_MORTON_VOXELLIST || type == MT_PROBAB_MORTON_VOXELLIST)
  {
    VoxelList<VOXELLIST_FLAGS_SIZE> *_voxellist = dynamic_cast<VoxelList<VOXELLIST_FLAGS_SIZE>*>(map);
    if (_voxellist == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'VoxelList' failed!" << endl);
    if(offset != Vector3ui())
      LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);

    num_collisions = this->template intersect<VOXELLIST_FLAGS_SIZE, true, false>(*_voxellist);
  }
  else if (type == MT_BITVECTOR_VOXELLIST || type == MT_PROBAB_VOXELLIST)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  else
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);

  return num_collisions;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypes(
    const GpuVoxelsMapSharedPtr other, voxelmap::BitVectorVoxel& types_in_collision, float coll_threshold, const Vector3ui &offset)
{
  size_t num_collisions = SSIZE_MAX;
  GpuVoxelsMap* map = other.get();
  MapType type = map->getMapType();

  if (type == MT_BITVECTOR_VOXELMAP)
  {
    voxelmap::BitVectorVoxelMap* _voxelmap = dynamic_cast<voxelmap::BitVectorVoxelMap*>(map);
    if (_voxelmap == NULL)
      LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelMap' failed!" << endl);

    num_collisions = this->template intersect_sparse<true, true, voxelmap::BitVectorVoxel>(*_voxelmap, &types_in_collision, 0, offset);
  }
  else
    LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);

  return num_collisions;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertRobotConfiguration(
    const MetaPointCloud *robot_links, bool with_self_collision_test)
{
  LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return false;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloud(
    const MetaPointCloud &meta_point_cloud, VoxelType voxelType)
{
  if (voxelType != 0)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_VT_0 << endl);

  // Get adress from device
  Vector3f* d_points = NULL;
  MetaPointCloudStruct tmp_struct;
  HANDLE_CUDA_ERROR(cudaMemcpy(&tmp_struct, meta_point_cloud.getDeviceConstPointer(), sizeof(MetaPointCloudStruct), cudaMemcpyDeviceToHost));
  HANDLE_CUDA_ERROR(cudaMemcpy(&d_points, tmp_struct.clouds_base_addresses, sizeof(Vector3f*), cudaMemcpyDeviceToHost));

  size_t num_points = meta_point_cloud.getAccumulatedPointcloudSize();
  thrust::device_vector<Vector3ui> d_voxels(num_points);
  kernel_toVoxels<<<this->numBlocks, this->numThreadsPerBlock>>>(d_points, num_points, D_PTR(d_voxels), this->m_resolution / 1000.0f);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  insertVoxelData(d_voxels);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloud(
    const MetaPointCloud& meta_point_cloud, const std::vector<VoxelType>& voxel_types)
{
  /* Basically this is a dummy implementation since the method can't be left abstract.
     However, I'm not sure whether this functionality makes sense here, so I didn't
     implement it.
   */
  LOGGING_WARNING_C(OctreeLog, NTree, "This functionality is not implemented, yet. The pointcloud will be inserted with the first VoxelType." << endl);
  insertMetaPointCloud(meta_point_cloud, voxel_types.front());
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
std::size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getMemoryUsage()
{
  return this->getMemUsage();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::clearMap()
{
  this->clear();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::clearVoxelType(VoxelType voxel_type)
{
  if (voxel_type != 0)
    LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_VT_0 << endl);
  else
    clearMap();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::writeToDisk(const std::string path)
{
  std::ofstream out(path.c_str());
  this->serialize(out);
  out.close();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::readFromDisk(const std::string path)
{
  try
  {
    std::ifstream in(path.c_str());
    this->deserialize(in);
    in.close();
    return true;
  } catch (std::ifstream::failure& e)
  {
    LOGGING_ERROR_C(OctreeLog, NTree, "DESERIALIZE FAILD!" << endl);
    return false;
  }
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild()
{
  return this->NTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild();
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::rebuild()
{
  this->NTree<branching_factor, level_count, InnerNode, LeafNode>::rebuild();
  return true;
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
Vector3ui GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getDimensions()
{
  uint32_t s = static_cast<uint32_t>(getVoxelSideLength<branching_factor>(level_count - 1));
  return Vector3ui(s, s, s);
}

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
Vector3f GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getMetricDimensions()
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
  if (this->m_has_data)
  {
    // Have to insert voxels and adjust occupancy since there are already some voxels in the NTree
    // Transform voxel coordinates to morton code
    thrust::device_vector<VoxelID> d_voxels_morton(num_points);
    kernel_toMortonCode<<<this->numBlocks, this->numThreadsPerBlock>>>(D_PTR(d_voxels), num_points,
    D_PTR(d_voxels_morton));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    // Sort and remove duplicates
    // TODO: remove thrust::unique() and adapt NTree::insert() to handle duplicates in the input data
    thrust::sort(d_voxels_morton.begin(), d_voxels_morton.end());
    thrust::device_vector<VoxelID>::iterator new_end = thrust::unique(d_voxels_morton.begin(),
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

}  // end of ns
}  // end of ns

#endif
