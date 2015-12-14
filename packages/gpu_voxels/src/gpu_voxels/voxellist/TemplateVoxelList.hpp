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
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------

#include "TemplateVoxelList.h"
#include <gpu_voxels/logging/logging_voxellist.h>
#include <gpu_voxels/voxellist/kernels/VoxelListOperations.hpp>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/system_error.h>

namespace gpu_voxels {
namespace voxellist {

template<class Voxel, class VoxelIDType>
TemplateVoxelList<Voxel, VoxelIDType>::TemplateVoxelList(const Vector3ui ref_map_dim, const float voxel_sidelength, const MapType map_type)
  : m_voxel_side_length(voxel_sidelength),
    m_ref_map_dim(ref_map_dim)
{
  this->m_map_type = map_type;

  m_collision_check_results = new bool[cMAX_NR_OF_BLOCKS];
  m_collision_check_results_counter = new uint16_t[cMAX_NR_OF_BLOCKS];

  // initialize result arrays
  for (uint32_t i = 0; i < cMAX_NR_OF_BLOCKS; i++)
  {
    m_collision_check_results[i] = false;
    m_collision_check_results_counter[i] = 0;
  }

  HANDLE_CUDA_ERROR(cudaMalloc((void** )&m_dev_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool)));
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_dev_collision_check_results_counter, cMAX_NR_OF_BLOCKS * sizeof(uint16_t)));

  // copy initialized arrays to device
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results, m_collision_check_results, cMAX_NR_OF_BLOCKS * sizeof(bool),
                 cudaMemcpyHostToDevice));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_collision_check_results_counter, m_collision_check_results_counter,
                 cMAX_NR_OF_BLOCKS * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

template<class Voxel, class VoxelIDType>
TemplateVoxelList<Voxel, VoxelIDType>::~TemplateVoxelList()
{
}


/*!
 * \brief TemplateVoxelList<Voxel>::make_unique
 * Sorting stuff and make unique
 * This has to be executed in that order, as "unique" only removes duplicates if they appear in a row!!
 * The keys and the values are sorted according to the key.
 * After sorting, the bitvectors of successive voxels with the same key are merged.
 * After that the keys are unified. According values are dropped too.
 */
template<class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::make_unique()
{
  // Sort all entries by key.
  try
  {
    LOGGING_DEBUG_C(VoxellistLog, TemplateVoxelList, "List size before make_unique: " << m_dev_list.size() << endl);

    // the ZipIterator represents the data that is sorted by the keys in m_dev_id_list
    thrust::sort_by_key(m_dev_id_list.begin(), m_dev_id_list.end(),
                        thrust::make_zip_iterator( thrust::make_tuple(m_dev_coord_list.begin(),m_dev_list.begin()) ),
                        thrust::less<VoxelIDType>());

  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception while sorting: " << e.what() << endl);
    exit(-1);
  }
  try
  {
    // Reverse iterate over sorted entries and merge successive voxel-bitvectors into the predecessor
    // of voxels with the same key. We dont touch the coordinates as they are the same either.
    thrust::inclusive_scan( thrust::make_reverse_iterator( thrust::make_zip_iterator( thrust::make_tuple(m_dev_id_list.end(), m_dev_list.end()) ) ),
                            thrust::make_reverse_iterator( thrust::make_zip_iterator( thrust::make_tuple(m_dev_id_list.begin(), m_dev_list.begin()) ) ),
                            thrust::make_reverse_iterator( thrust::make_zip_iterator( thrust::make_tuple(m_dev_id_list.end(), m_dev_list.end()) ) ),
                            Merge<Voxel, VoxelIDType>() );
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception while doing inclusive_scan: " << e.what() << endl);
    exit(-1);
  }
  // Now drop all duplicates.
  // This will remove successors and keep the first entry with the merged bitvectors.
  try
  {
    thrust::pair< keyIterator, zipValuesIterator > new_end;
    new_end = thrust::unique_by_key(m_dev_id_list.begin(), m_dev_id_list.end(),
                                    thrust::make_zip_iterator( thrust::make_tuple(m_dev_coord_list.begin(),m_dev_list.begin()) ) );

    size_t new_lenght = thrust::distance(m_dev_id_list.begin(), new_end.first);
    m_dev_id_list.resize(new_lenght);
    m_dev_coord_list.resize(new_lenght);
    m_dev_list.resize(new_lenght);
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception while dropping duplicates: " << e.what() << endl);
    exit(-1);
  }
  LOGGING_DEBUG_C(VoxellistLog, TemplateVoxelList, "List size after make_unique: " << m_dev_list.size() << endl);
}


template<class Voxel, class VoxelIDType>
size_t TemplateVoxelList<Voxel, VoxelIDType>::collideVoxellists(TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3ui &offset, thrust::device_vector<bool>& collision_stencil)
{
  bool locked_this = false;
  bool locked_other = false;
  uint32_t counter = 0;

  while (!locked_this && !locked_other)
  {
    // lock mutexes
    while (!locked_this)
    {
      locked_this = lockMutex();
      if(!locked_this) boost::this_thread::yield();
    }
    while (!locked_other && (counter < 50))
    {
      locked_other = other->lockMutex();
      if(!locked_other) boost::this_thread::yield();
      counter++;
    }
    if (!locked_other)
    {
      LOGGING_WARNING_C(VoxellistLog, TemplateVoxelList, "Could not lock other map since 50 trials!" << endl);
      counter = 0;
      unlockMutex();
      boost::this_thread::yield();
    }
  }


  // searching for the elements of "other" in "this". Therefore stencil has to be the size of "this"
  try
  {
    // if offset is given, we need our own comparison opperator!
    if(offset != Vector3ui(0))
    {
      LOGGING_WARNING_C(VoxellistLog, TemplateVoxelList, "Offset for VoxelList collision was given. Thrust performace is not optimal with that!" << endl);
      thrust::binary_search(thrust::device,
                            other->m_dev_id_list.begin(), other->m_dev_id_list.end(),
                            m_dev_id_list.begin(), m_dev_id_list.end(),
                            collision_stencil.begin(), offsetLessOperator<VoxelIDType>(m_ref_map_dim, offset));
    }else{
      thrust::binary_search(thrust::device,
                            other->m_dev_id_list.begin(), other->m_dev_id_list.end(),
                            m_dev_id_list.begin(), m_dev_id_list.end(),
                            collision_stencil.begin());
    }
    other->unlockMutex();
    unlockMutex();


    return thrust::count(collision_stencil.begin(), collision_stencil.end(), true);
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }
}

template<class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::insertPointCloud(const std::vector<Vector3f> &points, const BitVoxelMeaning voxel_meaning)
{

  uint32_t offset_new_entries = m_dev_list.size();

  // resize capacity
  m_dev_list.resize(offset_new_entries + points.size());
  m_dev_coord_list.resize(offset_new_entries + points.size());
  m_dev_id_list.resize(offset_new_entries + points.size());

  // get raw pointers to the thrust vectors data:
  Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
  Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

  // copy points to the gpu
  Vector3f* d_points;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_points, points.size() * sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(
        cudaMemcpy(d_points, &points[0], points.size() * sizeof(Vector3f), cudaMemcpyHostToDevice));

  uint32_t num_blocks, threads_per_block;
  computeLinearLoad(points.size(), &num_blocks, &threads_per_block);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelInsertGlobalPointCloud<<<num_blocks, threads_per_block>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
                                                                  m_ref_map_dim, m_voxel_side_length,
                                                                  d_points, points.size(), offset_new_entries, voxel_meaning);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaFree(d_points));

  make_unique();
}

template<class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, BitVoxelMeaning voxel_meaning)
{
  uint32_t total_points = meta_point_cloud.getAccumulatedPointcloudSize();

  uint32_t offset_new_entries = m_dev_list.size();
  // resize capacity
  m_dev_list.resize(offset_new_entries + total_points);
  m_dev_coord_list.resize(offset_new_entries + total_points);
  m_dev_id_list.resize(offset_new_entries + total_points);

  // get raw pointers to the thrust vectors data:
  Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
  Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

  computeLinearLoad(total_points, &m_blocks, &m_threads);
  kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
                                                      m_ref_map_dim, m_voxel_side_length,
                                                      meta_point_cloud.getDeviceConstPointer(),
                                                      offset_new_entries, voxel_meaning);
  make_unique();
}

template<class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::insertMetaPointCloud(const MetaPointCloud &meta_point_cloud,
                                                    const std::vector<BitVoxelMeaning>& voxel_meanings)
{
  if(meta_point_cloud.getNumberOfPointclouds() != voxel_meanings.size())
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Number of VoxelMeanings differs from number of Sub-Pointclouds! Not inserting MetaPointCloud!" << endl);
    return;
  }

  uint32_t total_points = meta_point_cloud.getAccumulatedPointcloudSize();

  uint32_t offset_new_entries = m_dev_list.size();
  // resize capacity
  m_dev_list.resize(offset_new_entries + total_points);
  m_dev_coord_list.resize(offset_new_entries + total_points);
  m_dev_id_list.resize(offset_new_entries + total_points);

  // get raw pointers to the thrust vectors data:
  Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
  Vector3ui* dev_coord_list_ptr = thrust::raw_pointer_cast(m_dev_coord_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

  BitVoxelMeaning* dev_voxel_meanings;
  size_t size = voxel_meanings.size() * sizeof(BitVoxelMeaning);
  HANDLE_CUDA_ERROR(cudaMalloc((void**) &dev_voxel_meanings, size));
  HANDLE_CUDA_ERROR(cudaMemcpy(dev_voxel_meanings, &voxel_meanings[0], size, cudaMemcpyHostToDevice));

  computeLinearLoad(total_points, &m_blocks, &m_threads);
  kernelInsertMetaPointCloud<<<m_blocks, m_threads>>>(dev_id_list_ptr, dev_coord_list_ptr, dev_voxel_list_ptr,
                                                      m_ref_map_dim, m_voxel_side_length,
                                                      meta_point_cloud.getDeviceConstPointer(),
                                                      offset_new_entries, dev_voxel_meanings);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaFree(dev_voxel_meanings));

  make_unique();
}

template<class Voxel, class VoxelIDType>
size_t TemplateVoxelList<Voxel, VoxelIDType>::collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold,
                                             const Vector3ui &offset)
{
  size_t collisions = SSIZE_MAX;
  switch (other->getMapType())
  {
    case MT_PROBAB_VOXELMAP:
    {
      LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      break;
    }
    case MT_BITVECTOR_VOXELLIST:
    {
      BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>* m = (BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>*) other.get();
      thrust::device_vector<bool> collision_stencil(m_dev_id_list.size()); // Temporary data structure
      collisions = collideVoxellists(m, offset, collision_stencil);
      break;
    }
    case MT_BITVECTOR_VOXELMAP:
    {
      DefaultCollider collider(coll_threshold);
      voxelmap::BitVectorVoxelMap* m = (voxelmap::BitVectorVoxelMap*) other.get();
      collisions = collisionCheckWithCollider(m, collider, offset);
      break;
    }
    case MT_BITVECTOR_OCTREE:
    {
      LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << " " << GPU_VOXELS_MAP_SWAP_FOR_COLLIDE << endl);
      break;
    }
    default:
    {
      LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      break;
    }
  }
  return collisions;
}

template<class Voxel, class VoxelIDType>
size_t TemplateVoxelList<Voxel, VoxelIDType>::collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold,
                                                       const uint32_t resolution_level,
                                                       const Vector3ui &offset)
{
  LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
  return SSIZE_MAX;
}

template<class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::shrinkToFit()
{
  m_dev_list.shrink_to_fit();
  m_dev_coord_list.shrink_to_fit();
  m_dev_id_list.shrink_to_fit();
}

template<class Voxel, class VoxelIDType>
size_t TemplateVoxelList<Voxel, VoxelIDType>::getMemoryUsage() const
{
  return (m_dev_list.size() * sizeof(Voxel) +
          m_dev_coord_list.size() * sizeof(Vector3ui) +
          m_dev_id_list.size() * sizeof(VoxelIDType));
}

template<class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::clearMap()
{
  while (!lockMutex())
  {
    boost::this_thread::yield();
  }
  m_dev_list.clear();
  m_dev_coord_list.clear();
  m_dev_id_list.clear();
  unlockMutex();
}

template<class Voxel, class VoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::writeToDisk(const std::string path)
{
  LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Dumping VoxelList to disk: " <<
                 getDimensions().x << " Voxels ==> " << (getMemoryUsage() / 1024.0 / 1024.0) << " MB. ..." << endl);

  std::ofstream out(path.c_str());

  if(!out.is_open())
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Write to file " << path << " failed!" << endl);
    return false;
  }
  thrust::host_vector<VoxelIDType> host_id_list = m_dev_id_list;
  thrust::host_vector<Vector3ui> host_coord_list = m_dev_coord_list;
  thrust::host_vector<Voxel> host_list = m_dev_list;

  uint32_t num_voxels = host_list.size();
  int32_t map_type = m_map_type;

  out.write((char*) &map_type, sizeof(int32_t));
  out.write((char*) &m_ref_map_dim, sizeof(Vector3ui));
  out.write((char*) &m_voxel_side_length, sizeof(float));
  out.write((char*) &num_voxels, sizeof(uint32_t));
  out.write((char*) &host_id_list[0], num_voxels * sizeof(VoxelIDType));
  out.write((char*) &host_coord_list[0], num_voxels * sizeof(Vector3ui));
  out.write((char*) &host_list[0], num_voxels * sizeof(Voxel));

  out.close();
  LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Write to disk done: Extracted "<< num_voxels << " Voxels." << endl);
  return true;
}

template<class Voxel, class VoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::readFromDisk(const std::string path)
{

  thrust::host_vector<VoxelIDType> host_id_list;
  thrust::host_vector<Vector3ui> host_coord_list;
  thrust::host_vector<Voxel> host_list;

  uint32_t num_voxels;
  float voxel_side_lenght;
  Vector3ui ref_map_dim;
  int32_t map_type;

  std::ifstream in(path.c_str());
  if(!in.is_open())
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file " << path << " failed!"<< endl);
    return false;
  }

  in.read((char*) &map_type, sizeof(int32_t));
  if(map_type != m_map_type)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file failed: The map type does not match current object!" << endl);
    return false;
  }
  in.read((char*)&ref_map_dim, sizeof(Vector3ui));
  if(ref_map_dim != m_ref_map_dim)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file failed: Read reference map dimension does not match current object!" << endl);
    return false;
  }
  in.read((char*)&voxel_side_lenght, sizeof(float));
  if(voxel_side_lenght != m_voxel_side_length)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Read from file failed: Read Voxel side lenght does not match current object!" << endl);
    return false;
  }
  in.read((char*)&num_voxels, sizeof(uint32_t));

  host_id_list.resize(num_voxels);
  host_coord_list.resize(num_voxels);
  host_list.resize(num_voxels);
  in.read((char*) &host_id_list[0], num_voxels * sizeof(VoxelIDType));
  in.read((char*) &host_coord_list[0], num_voxels * sizeof(Vector3ui));
  in.read((char*) &host_list[0], num_voxels * sizeof(Voxel));

  in.close();
  LOGGING_INFO_C(VoxellistLog, TemplateVoxelList, "Read "<< num_voxels << " Voxels from file." << endl;);


  m_dev_id_list = host_id_list;
  m_dev_coord_list = host_coord_list;
  m_dev_list = host_list;

  return true;
}

template<class Voxel, class VoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::merge(const GpuVoxelsMapSharedPtr other, const Vector3ui &voxel_offset, const BitVoxelMeaning *new_meaning)
{
  switch (other->getMapType())
  {
    case MT_BITVECTOR_VOXELLIST:
    {
      BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>* m = (BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>*) other.get();

      uint32_t num_new_voxels = m->getDimensions().x;
      uint32_t offset_new_entries = m_dev_list.size();
      // resize capacity
      m_dev_list.resize(offset_new_entries + num_new_voxels);
      m_dev_coord_list.resize(offset_new_entries + num_new_voxels);
      m_dev_id_list.resize(offset_new_entries + num_new_voxels);

      // We append the given list to our own list of points.
      thrust::copy(
        thrust::make_zip_iterator( thrust::make_tuple(m->m_dev_list.begin(), m->m_dev_coord_list.begin(), m->m_dev_id_list.begin()) ),
        thrust::make_zip_iterator( thrust::make_tuple(m->m_dev_list.end(),   m->m_dev_coord_list.end(),   m->m_dev_id_list.end()) ),
        thrust::make_zip_iterator( thrust::make_tuple(m_dev_list.begin()+offset_new_entries,    m_dev_coord_list.begin()+offset_new_entries,    m_dev_id_list.begin()+offset_new_entries) ) );

      // If an offset was given, we have to alter the newly added voxels.
      if(voxel_offset != Vector3ui())
      {
        thrust::transform(
          thrust::make_zip_iterator( thrust::make_tuple(m_dev_coord_list.begin()+offset_new_entries, m_dev_id_list.begin()+offset_new_entries) ),
          thrust::make_zip_iterator( thrust::make_tuple(m_dev_coord_list.end(),   m_dev_id_list.end()) ),
          thrust::make_zip_iterator( thrust::make_tuple(m_dev_coord_list.begin()+offset_new_entries, m_dev_id_list.begin()+offset_new_entries) ),
          applyOffsetOperator<VoxelIDType>(m_ref_map_dim, voxel_offset));
      }

      // if a new meaning was given, iterate over the voxellist and overwrite the meaning
      if(new_meaning)
      {
        BitVectorVoxel fillVoxel;
        fillVoxel.bitVector().setBit(*new_meaning);
        thrust::fill(m_dev_list.begin()+offset_new_entries, m_dev_list.end(), fillVoxel);
      }

      make_unique();

      return true;
    }
    default:
    {
      LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      return false;
    }
  }
}

template<class Voxel, class VoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::merge(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset, const BitVoxelMeaning *new_meaning)
{
  Vector3ui voxel_offset = voxelmap::mapToVoxels(m_voxel_side_length, metric_offset);
  return merge(other, voxel_offset, new_meaning);
}

template<class Voxel, class VoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::subtract(const GpuVoxelsMapSharedPtr other, const Vector3ui &voxel_offset)
{
  switch (other->getMapType())
  {
    case MT_BITVECTOR_VOXELLIST:
    {
      BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>* m = (BitVoxelList<BIT_VECTOR_LENGTH, VoxelIDType>*) other.get();

      // find the overlapping voxels:
      thrust::device_vector<bool> overlap_stencil(m_dev_id_list.size()); // A stencil of the voxels in collision
      size_t overlapping_voxels = collideVoxellists(m, voxel_offset, overlap_stencil);

      keyCoordVoxelZipIterator new_end;

      // remove the overlapping voxels:
      new_end = thrust::remove_if(thrust::make_zip_iterator( thrust::make_tuple(m_dev_id_list.begin(), m_dev_coord_list.begin(), m_dev_list.begin()) ),
                                  thrust::make_zip_iterator( thrust::make_tuple(m_dev_id_list.end(), m_dev_coord_list.end(), m_dev_list.end()) ),
                                  overlap_stencil.begin(),
                                  thrust::identity<bool>());

      size_t new_lenght = thrust::distance(m_dev_id_list.begin(), thrust::get<0>(new_end.get_iterator_tuple()));
      m_dev_id_list.resize(new_lenght);
      m_dev_coord_list.resize(new_lenght);
      m_dev_list.resize(new_lenght);

      return true;
    }
    default:
    {
      LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
      return false;
    }
  }
}

template<class Voxel, class VoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::subtract(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset)
{
  Vector3ui voxel_offset = voxelmap::mapToVoxels(m_voxel_side_length, metric_offset);
  return subtract(other, voxel_offset);
}


template<class Voxel, class VoxelIDType>
Vector3ui TemplateVoxelList<Voxel, VoxelIDType>::getDimensions() const
{
  //LOGGING_WARNING_C(VoxellistLog, TemplateVoxelList, "This is not the xyz dimension! The x value contains the number of voxels in the list." << endl);
  return Vector3ui(m_dev_list.size(), 0, 0);
}

template<class Voxel, class VoxelIDType>
Vector3f TemplateVoxelList<Voxel, VoxelIDType>::getMetricDimensions() const
{
  return Vector3f(m_ref_map_dim.x, m_ref_map_dim.y, m_ref_map_dim.z) * getVoxelSideLength();
}


template <class Voxel, class VoxelIDType>
void TemplateVoxelList<Voxel, VoxelIDType>::extractCubes(thrust::device_vector<Cube>** output_vector) const
{
  try
  {
    if (*output_vector == NULL)
    {
      *output_vector = new thrust::device_vector<Cube>(m_dev_list.size());
    }
    else
    {
      (*output_vector)->resize(m_dev_list.size());
    }
    // Transform Iterator that takes coordinates and bitvector and writes cubes to output_vector
    thrust::transform(m_dev_coord_list.begin(), m_dev_coord_list.end(), m_dev_list.begin(), (*output_vector)->begin(),
                      VoxelToCube());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  }
  catch(thrust::system_error &e)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, "Caught Thrust exception " << e.what() << endl);
    exit(-1);
  }
}


template <class Voxel, class VoxelIDType>
template<class OtherVoxel, class Collider>
size_t TemplateVoxelList<Voxel, VoxelIDType>::collisionCheckWithCollider(voxelmap::TemplateVoxelMap<OtherVoxel>* other,
                                                              Collider collider, const Vector3ui& offset)
{
  // Map Dims have to be equal to be able to compare pointer adresses!
  if(other->getDimensions() != m_ref_map_dim)
  {
    LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList,
                    "The dimensions of the Voxellist reference map do not match the colliding voxel map dimensions. Not checking collisions!" << endl);
    return SSIZE_MAX;
  }


  bool locked_this = false;
  bool locked_other = false;
  uint32_t counter = 0;

  uint32_t number_of_collisions = 0;
  while (!locked_this && !locked_other)
  {
    // lock mutexes
    while (!locked_this)
    {
      locked_this = lockMutex();
      if(!locked_this) boost::this_thread::yield();
    }
    while (!locked_other && (counter < 50))
    {
      locked_other = other->lockMutex();
      if(!locked_other) boost::this_thread::yield();
      counter++;
    }
    if (!locked_other)
    {
      LOGGING_WARNING_C(VoxellistLog, TemplateVoxelList, "Could not lock other map since 50 trials!" << endl);
      counter = 0;
      unlockMutex();
      boost::this_thread::yield();
    }
  }
  // get raw pointers to the thrust vectors data:
  Voxel* dev_voxel_list_ptr = thrust::raw_pointer_cast(m_dev_list.data());
  VoxelIDType* dev_id_list_ptr = thrust::raw_pointer_cast(m_dev_id_list.data());

  uint32_t num_blocks, threads_per_block;
  computeLinearLoad(getDimensions().x, &num_blocks, &threads_per_block);
  size_t dynamic_shared_mem_size = sizeof(BitVectorVoxel) * cMAX_THREADS_PER_BLOCK;

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  kernelCollideWithVoxelMap<<<num_blocks, threads_per_block, dynamic_shared_mem_size>>>(dev_id_list_ptr, dev_voxel_list_ptr, getDimensions().x,
                                                               other->getDeviceDataPtr(), m_ref_map_dim, collider, offset,
                                                               m_dev_collision_check_results_counter);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_collision_check_results_counter, m_dev_collision_check_results_counter,
                 cMAX_NR_OF_BLOCKS * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  for (uint32_t i = 0; i < num_blocks; i++)
  {
    number_of_collisions += m_collision_check_results_counter[i];
  }


  other->unlockMutex();
  unlockMutex();

  return number_of_collisions;
}


template <class Voxel, class VoxelIDType>
template<class OtherVoxel, class OtherVoxelIDType>
bool TemplateVoxelList<Voxel, VoxelIDType>::equals(const TemplateVoxelList<OtherVoxel, OtherVoxelIDType>& other) const
{
  if((m_ref_map_dim != other.m_ref_map_dim) ||
     (m_voxel_side_length != other.m_voxel_side_length) ||
     (getDimensions()) != other.getDimensions())
  {
    return false;
  }

  bool equal = true;
  equal &= thrust::equal(m_dev_list.begin(), m_dev_list.end(), other.m_dev_list.begin());
  equal &= thrust::equal(m_dev_id_list.begin(), m_dev_id_list.end(), other.m_dev_id_list.begin());
  equal &= thrust::equal(m_dev_coord_list.begin(), m_dev_coord_list.end(), other.m_dev_coord_list.begin());
  return equal;
}

} // end of namespace voxellist
} // end of namespace gpu_voxels
