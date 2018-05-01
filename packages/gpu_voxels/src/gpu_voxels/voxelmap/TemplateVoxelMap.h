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
 * \date    2014-07-10
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_TEMPLATE_VOXELMAP_H_INCLUDED

#include <vector>
#include <boost/thread.hpp>
#include <cuda_runtime.h>

#include <gpu_voxels/helpers/cuda_handling.hpp>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/voxelmap/AbstractVoxelMap.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>
#include <gpu_voxels/voxel/DefaultCollider.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
namespace voxelmap {

template<class Voxel>
class TemplateVoxelMap : public AbstractVoxelMap
{
public:
  /*! Create a voxelmap that holds dim.x * dim.y * dim.z voxels.
   *  A voxel is treated as cube with side length voxel_side_length. */
  TemplateVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type);

  /*!
   * This constructor does NOT create a new voxel map on the GPU.
   * The new object will represent the voxel map specified in /p dev_data.
   * Warning: Not all member variables will be set correctly for the map.
   */
  TemplateVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type);

  //! Destructor
  virtual ~TemplateVoxelMap();

  /* ======== getter functions ======== */

  //! get pointer to data array on device
  Voxel* getDeviceDataPtr()
  {
    return m_dev_data;
  }

  const Voxel* getConstDeviceDataPtr() const
  {
    return m_dev_data;
  }

  inline virtual void* getVoidDeviceDataPtr()
  {
    return (void*) m_dev_data;
  }

  inline virtual const void* getConstVoidDeviceDataPtr() const
  {
    return (const void*) m_dev_data;
  }

  //! get the number of voxels held in the voxelmap
  inline uint32_t getVoxelMapSize() const
  {
    return m_dim.x * m_dim.y * m_dim.z;
  }

  //! get the side length of the voxels.
  inline virtual float getVoxelSideLength() const
  {
    return m_voxel_side_length;
  }

  /* ======== VoxelMap operations  ======== */
  /*! as above, without locking mutex, which then must be done manually!
   * This might be necessary for combination with other operations to ensure
   * that the map did not change since it was cleared.
   */
  void clearVoxelMapRemoteLock(BitVoxelMeaning voxel_meaning);

  //! print data array to screen for debugging (low performance)
  virtual void printVoxelMapData();

  virtual void gatherVoxelsByIndex(thrust::device_ptr<uint> dev_indices_begin, thrust::device_ptr<uint> dev_indices_end, thrust::device_ptr<Voxel> dev_output_begin);

  /* --- collision check operations --- */
  /*! Test for collision with other VoxelMap
   *  with given occupancy thresholds.
   *  Returns true if there is any collision.
   *
   *  Assumes same dimensions and voxel_side_length
   *  as local VoxelMap. See also getDimensions() function.
   */
  template< class OtherVoxel, class Collider>
  bool collisionCheck(TemplateVoxelMap<OtherVoxel>* other, Collider collider);


//  __host__
//  bool collisionCheckAlternative(const uint8_t threshold, VoxelMap* other,
//          const uint8_t other_threshold, uint32_t loop_size);

  template< class OtherVoxel, class Collider>
  uint32_t collisionCheckWithCounter(TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider());

  template< class OtherVoxel, class Collider>
  uint32_t collisionCheckWithCounterRelativeTransform(TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider(), const Vector3i &offset = Vector3i());

//  __host__
//  bool collisionCheckBoundingBox(uint8_t threshold, VoxelMap* other, uint8_t other_threshold,
//                        Vector3ui bounding_box_start, Vector3ui bounding_box_end);


  // ------ BEGIN Global API functions ------
  virtual void insertPointCloud(const std::vector<Vector3f> &point_cloud, const BitVoxelMeaning voxelmeaning);

  virtual void insertPointCloud(const PointCloud &pointcloud, const BitVoxelMeaning voxel_meaning);

  virtual void insertPointCloud(const Vector3f* points_d, uint32_t size, const BitVoxelMeaning voxel_meaning);

  /**
   * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map.
   * @param meta_point_cloud The MetaPointCloud to insert
   * @param voxel_meaning Voxel meaning of all voxels
   */
  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, BitVoxelMeaning voxel_meaning);

  /**
   * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map. Each pointcloud
   * inside the MetaPointCloud will get it's own voxel meaning as given in the voxel_meanings
   * parameter. The number of pointclouds in the MetaPointCloud and the size of voxel_meanings
   * have to be identical.
   * @param meta_point_cloud The MetaPointCloud to insert
   * @param voxel_meanings Vector with voxel meanings
   */
  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings);

  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset = Vector3f(), const BitVoxelMeaning* new_meaning = NULL);
  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3i &voxel_offset = Vector3i(), const BitVoxelMeaning* new_meaning = NULL);

  virtual std::size_t getMemoryUsage() const
  {
    return m_dim.x * m_dim.y * m_dim.z * sizeof(Voxel);
  }

  virtual void clearMap();
  //! set voxel occupancies for a specific voxelmeaning to zero

  virtual bool writeToDisk(const std::string path);

  virtual bool readFromDisk(const std::string path);

  virtual Vector3ui getDimensions() const;

  virtual Vector3f getMetricDimensions() const;

  // ------ END Global API functions ------


protected:

  /* ======== Variables with content on host ======== */
  const Vector3ui m_dim;
  const Vector3f m_limits;
  float m_voxel_side_length;
  uint32_t m_voxelmap_size;
  //uint32_t m_num_points;
  uint32_t m_blocks;
  uint32_t m_threads;
  uint32_t m_alternative_blocks;
  uint32_t m_alternative_threads;

  //! size of array for collision check
  uint32_t m_result_array_size;

  //! result array for collision check
  bool* m_collision_check_results;
  //! result array for collision check with counter
  uint16_t* m_collision_check_results_counter;

  //! performance measurement start time
  cudaEvent_t m_start;
  //! performance measurement stop time
  cudaEvent_t m_stop;
  //! performance measurement elapsed time
  float m_elapsed_time;

  /* ======== Variables with content on device ======== */

  /*! VoxelMap data on device.
   *  storage format is: index = z * dim.x * dim.y + y * dim.x + x  */
  Voxel* m_dev_data;
  
  /*! This is used by insertion kernels to indicate,
   * if points were outside map dimensions 
   * and could not be inserted */
  bool* m_dev_points_outside_map;

  /* some variables are mirrored on device to reduce
   * copy overhead when access from kernels is necessary  */

  //! results of collision check on device
  bool* m_dev_collision_check_results;

  //! result array for collision check with counter on device
  uint16_t* m_dev_collision_check_results_counter;

};

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
