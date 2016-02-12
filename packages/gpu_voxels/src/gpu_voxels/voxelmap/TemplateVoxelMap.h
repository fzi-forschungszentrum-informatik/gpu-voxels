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
//#include <gpu_voxels/voxelmap/Voxel.h>
#include <gpu_voxels/helpers/CudaMath.h>
#include <gpu_voxels/helpers/MetaPointCloud.h>
#include <gpu_voxels/voxelmap/AbstractVoxelMap.h>
#include <gpu_voxels/voxel/DefaultCollider.h>
#include <gpu_voxels/voxelmap/kernels/VoxelMapOperations.h>



/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
namespace voxelmap {

//// todo: use VoxelMapConfig instead of the many other variables
//class VoxelMapConfig
//{
//public:
//  //! Constructor
//  __host__ VoxelMapConfig(uint32_t _dim_x, uint32_t _dim_y, uint32_t _dim_z, float _voxel_side_length) :
//      dimension(Vector3ui(_dim_x, _dim_y, _dim_z)), voxel_side_length(_voxel_side_length)
//  {
//  }
//
//  //! Copy constructor
//  __host__ VoxelMapConfig(const VoxelMapConfig& other) :
//      dimension(other.dimension), voxel_side_length(other.voxel_side_length), voxelmap(other.voxelmap)
//  {
//  }
//
//  //! Destructor
//  __host__ ~VoxelMapConfig()
//  {
//  }
//
//  Vector3ui dimension;
//  float voxel_side_length;
//  Voxel* voxelmap;
//};

template<class Voxel>
class TemplateVoxelMap : public AbstractVoxelMap
{
public:
  /*! Create a voxelmap that holds dim_x * dim_y * dim_z voxels.
   *  A voxel is treated as cube with side length voxel_side_length. */
  TemplateVoxelMap(const uint32_t dim_x, const uint32_t dim_y, const uint32_t dim_z, const float voxel_side_length, const MapType map_type);

  // __host__
  // VoxelMap(VoxelMapConfig config);
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

  //! gets an offset in pointer arithmetics
  ptrdiff_t getVoxelPtrOffset(const Vector3ui &coordinates)
  {
    return (coordinates.z * m_dim.x * m_dim.y + coordinates.y * m_dim.x + coordinates.x);
  }

  //! gets an offset in pointer arithmetics
  ptrdiff_t getVoxelPtrOffset(uint32_t x, uint32_t y, uint32_t z)
  {
    return (z * m_dim.x * m_dim.y + y * m_dim.x + x);
  }

  //! get pointer to specific voxel on device given the coordinates
  Voxel* getDeviceVoxelPtr(uint32_t x, uint32_t y, uint32_t z)
  {
    return (Voxel*) (intptr_t)(m_dev_data + (z * m_dim.x * m_dim.y + y * m_dim.x + x));
  }

  //! get pointer to specific voxel on device given the coordinates
  Voxel* getDeviceVoxelPtr(const Vector3ui &coordinates)
  {
    return (Voxel*) (intptr_t)(m_dev_data +
                     (coordinates.z * m_dim.x * m_dim.y + coordinates.y * m_dim.x + coordinates.x));
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
//  //! use a kernel call and print data from within
//  __host__
//  void printVoxelMapDataFromDevice();
  //! print data array to screen for debugging (low performance)
  virtual void printVoxelMapData();

//  //! write log for data / performance measurement
//  __host__
//  bool writeLog(std::string filename, uint32_t loop_size = 1, bool reset_values = false);

  /* ----- mutex locking and unlocking ----- */
  mutable boost::mutex m_mutex;
  bool lockMutex();

  void unlockMutex();

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
//  bool collisionCheckIndices(uint8_t threshold, uint32_t* index_list, uint32_t index_size);

//  __host__
//  uint64_t collisionCheckIndicesBitmap(uint8_t threshold,
//      uint32_t* index_list, uint32_t index_size, uint64_t* bitmap_list, int64_t offset_x, int64_t offset_y);
//
//
//  __host__
//  bool collisionCheckAlternative(const uint8_t threshold, VoxelMap* other,
//          const uint8_t other_threshold, uint32_t loop_size);

  template< class OtherVoxel, class Collider>
  uint32_t collisionCheckWithCounter(TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider());

  template< class OtherVoxel, class Collider>
  uint32_t collisionCheckWithCounterRelativeTransform(TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider(), const Vector3ui &offset = Vector3ui());

//  __host__
//  bool collisionCheckBoundingBox(uint8_t threshold, VoxelMap* other, uint8_t other_threshold,
//                        Vector3ui bounding_box_start, Vector3ui bounding_box_end);

  /* ======== some functions for self testing ======== */

//  __host__
//  bool triggerVoxelMapCollisionTestNoCollision(uint32_t num_data_points, TemplateVoxelMap<OtherVoxel>* other);
//  __host__
//  bool triggerVoxelMapCollisionTestWithCollision(uint32_t num_data_points, TemplateVoxelMap<OtherVoxel>* other);
//  __host__
//  bool triggerVoxelMapAddresSchemeTest(uint32_t nr_of_tests);
    /*==================================================*/
//  __host__
//   void insertBox(Vector3f cartesian_from, Vector3f cartesian_to,
//                  BitVoxelMeaning voxelmeaning, uint8_t occupancy = 255);

//   __host__
//   void insertBoxByIndices(Vector3ui indices_from, Vector3ui indices_to,
//                           BitVoxelMeaning voxelmeaning, uint8_t occupancy = 255);

//   __host__
//   float getVoxelSideLength();

//   __host__
//   void copyVoxelVectorToDevice(uint32_t index_list, uint32_t size, uint32_t* dev_voxel_list);

//   __host__
//   void insertVoxelVector(uint32_t* dev_voxel_list, uint32_t size, bool with_bitvector, uint64_t mask);

//   void insertBitmapByIndices(uint32_t size, uint32_t* index_list, uint64_t* bitmaps);


  // ------ BEGIN Global API functions ------
  virtual void insertPointCloud(const std::vector<Vector3f> &point_cloud, BitVoxelMeaning voxelmeaning);

  /**
   * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map.
   * @param meta_point_cloud The MetaPointCloud to insert
   * @param voxel_meaning Voxel meaning of all voxels
   */
  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, BitVoxelMeaning voxel_meaning);

  virtual size_t collideWith(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const Vector3ui &offset = Vector3ui());
  /**
   * @brief insertMetaPointCloud Inserts a MetaPointCloud into the map. Each pointcloud
   * inside the MetaPointCloud will get it's own voxel meaning as given in the voxel_meanings
   * parameter. The number of pointclouds in the MetaPointCloud and the size of voxel_meanings
   * have to be identical.
   * @param meta_point_cloud The MetaPointCloud to insert
   * @param voxel_meanings Vector with voxel meanings
   */
  virtual void insertMetaPointCloud(const MetaPointCloud &meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings);

  virtual size_t collideWithResolution(const GpuVoxelsMapSharedPtr other, float coll_threshold = 1.0, const uint32_t resolution_level = 0, const Vector3ui &offset = Vector3ui());

  virtual size_t collideWithTypes(const GpuVoxelsMapSharedPtr other, BitVectorVoxel&  meanings_in_collision, float coll_threshold = 1.0, const Vector3ui &offset = Vector3ui()) = 0;

  virtual size_t collideWithBitcheck(const GpuVoxelsMapSharedPtr other, const u_int8_t margin = 0, const Vector3ui &offset = Vector3ui());

  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset = Vector3f(), const BitVoxelMeaning* new_meaning = NULL);
  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3ui &voxel_offset = Vector3ui(), const BitVoxelMeaning* new_meaning = NULL);

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

   //------------------- Env Map specific functions: -------------------
   void initSensorSettings(const Sensor& sensor);

   void updateSensorPose(const Sensor& sensor);

   /*! Copies a sensor pointcloud to the device.
    * No further processing is done by this function!
    * Call transformSensorData() afterwards.
    */
   void copySensorDataToDevice(const Vector3f* points);

   /*! Transforms pointcloud according to the previously set
    * sensor position
    */
   void transformSensorData();

//   __host__
//   uint32_t getUpdateCounter();
//   // debugging.. todo: make this private again!
//   void increaseUpdateCounter();
   // ------------------- END of Env Map specific functions -------------------

   // ------------------- BEGIN Robot Map specific functions -------------------
//   __host__
//   void insertData(const Vector3f* points, const uint32_t num_points);

//   /*! Insert robot configuration. May check for self collision. See
//    *  also setSelfCollisionDependencies().
//    *  Returns FALSE if there is a collision
//    */
//   __host__
//   bool insertRobotConfiguration(const MetaPointCloud *robot_links,
//                            bool with_self_collision_test);


//   __host__
//   void insertConfigurationOverwritingSensorData(const MetaPointCloud *robot_links,
//                                                 VoxelMap* env_map);
//
//   __host__
//   void insertSweptVolumeConfiguration(uint32_t kinematic_chain_size, uint32_t* point_cloud_sizes,
//                                       uint32_t* dev_point_cloud_sizes, Vector3f** dev_point_clouds,
//                                       uint8_t swept_volume_index);
//   __host__
//   void removeSweptVolumeConfiguration(uint32_t kinematic_chain_size, uint32_t* point_cloud_sizes,
//                                       uint32_t* dev_point_cloud_sizes, Vector3f** dev_point_clouds,
//                                       uint8_t swept_volume_index);


//   /*! Model kinematic links that may not be entered into VoxelMap without
//       performing a self collision check.
//
//       Example assuming:
//          - the kinematic chain has a length of 6 links
//          - the links 0, 1 and 2 can not collide with each other
//          - link 3 can collide with one of the previous (does not matter with which)
//          - link 4 can collide with one of the previous
//          - link 5 can collide with one of the previous
//
//         In this case there is no need to insert every link after each other
//         and check for collision every time.
//
//         A valid order to insert the links would be:
//         0 (no check),
//         1 (no check),
//         2 (no check),
//         3 (self-collision check!),
//         4 (self-collision check!),
//         5 (self-collision check!)
//
//         This would be represented in the following matter:
//         links_to_enable_check = (3, 4, 5)
//
//         IMPORTANT: start to count with 0 and pay attention to correct order!       */
//   __host__
//   void setSelfCollisionDependencies(std::vector<uint32_t>& links_to_enable_check);
//
//   __host__
//   const std::vector<uint32_t>& getSelfCollisionDependencies() const
//   {
//     return m_links_to_enable_selfcol_check;
//   }

//   __host__
//   void clearBitvector(uint8_t bit_number);

//   __host__
//   void insertConfigurationIntoBitVector(uint32_t kinematic_chain_size, uint32_t* point_cloud_sizes,
//                                         uint32_t* dev_point_cloud_sizes, Vector3f** dev_point_clouds,
//                                         bool with_self_collision_test, const uint8_t bit_number);

   // ------------------- END of Robot Map specific functions -------------------

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

  /* some variables are mirrored on device to reduce
   * copy overhead when access from kernels is necessary  */

  //! results of collision check on device
  bool* m_dev_collision_check_results;

  //! result array for collision check with counter on device
  uint16_t* m_dev_collision_check_results_counter;

  //!  storage for measured data
  std::vector<float> m_measured_data;

  // ------------------- BEGIN Env Map specific: -------------------
  /* ======== Variables with content on host ======== */

  uint32_t m_blocks_sensor_operations;
  uint32_t m_threads_sensor_operations;

  bool m_init_sensor;
  Sensor m_sensor;


  /* ======== Variables with content on device ======== */

  //! device pointer to sensor parameters
  Sensor* m_dev_sensor;

  /*! device array for raw (untransformed) sensor
   *  data */
  Vector3f* m_dev_raw_sensor_data;

  //! device array for transformed sensor data
  Vector3f* m_dev_transformed_sensor_data;

  // ------------------- BEGIN Env map specific: -------------------
  uint32_t m_update_counter;
  // ------------------- END Env map specific -------------------

  // ------------------- BEGIN Robot map specific: -------------------
  uint32_t m_blocks_robot_operations;
  uint32_t m_threads_robot_operations;

  // array for point cloud data
  Vector3f* m_dev_point_data;

  // ----- self collision checking -----
  std::vector<uint32_t> m_links_to_enable_selfcol_check;
  bool m_selfcol_dependency_set;

  bool m_self_collision;
  bool* m_dev_self_collision;

//  void syncSelfCollisionInfoToHost();
//  void syncSelfCollisionInfoToDevice();
  // ------------------- END Robot map specific -------------------
};

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif
