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

#ifndef GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H
#define GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H

#include <boost/thread.hpp>

#include <gpu_voxels/voxellist/AbstractVoxelList.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

#include <gpu_voxels/voxel/DefaultCollider.h>
#include <gpu_voxels/voxelmap/BitVoxelMap.h>
#include <gpu_voxels/voxelmap/ProbVoxelMap.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


/**
 * @namespace gpu_voxels::voxelmap
 * Contains implementation of VoxelMap Datastructure and according operations
 */
namespace gpu_voxels {
namespace voxellist {

template<class Voxel, class VoxelIDType>
class TemplateVoxelList : public AbstractVoxelList
{
  typedef typename thrust::device_vector<VoxelIDType>::iterator  keyIterator;

  typedef typename thrust::device_vector<Vector3ui>::iterator  coordIterator;
  typedef typename thrust::device_vector<Voxel>::iterator  voxelIterator;

  typedef thrust::tuple<coordIterator, voxelIterator> valuesIteratorTuple;
  typedef thrust::zip_iterator<valuesIteratorTuple> zipValuesIterator;

  typedef thrust::tuple<keyIterator, voxelIterator> keyVoxelIteratorTuple;
  typedef thrust::zip_iterator<keyVoxelIteratorTuple> keyVoxelZipIterator;

public:
  typedef thrust::tuple<keyIterator, coordIterator, voxelIterator> keyCoordVoxelIteratorTriple;
  typedef thrust::zip_iterator<keyCoordVoxelIteratorTriple> keyCoordVoxelZipIterator;

  TemplateVoxelList(const Vector3ui ref_map_dim, const float voxel_side_length, const MapType map_type);

  //! Destructor
  virtual ~TemplateVoxelList();

  /* ======== getter functions ======== */

  //! get thrust triple to the beggining of all data vectors
  virtual  keyCoordVoxelZipIterator getBeginTripleZipIterator();

  //! get thrust triple to the end of all data vectors
  virtual keyCoordVoxelZipIterator getEndTripleZipIterator();

  //! get access to data vectors on device
  typename thrust::device_vector<Voxel>::iterator getDeviceDataVectorBeginning()
  {
    return m_dev_list.begin();
  }
  typename thrust::device_vector<VoxelIDType>::iterator getDeviceIdVectorBeginning()
  {
    return m_dev_id_list.begin();
  }
  typename thrust::device_vector<Vector3ui>::iterator getDeviceCoordVectorBeginning()
  {
    return m_dev_coord_list.begin();
  }

  //! get pointer to data array on device
  Voxel* getDeviceDataPtr()
  {
    return thrust::raw_pointer_cast(m_dev_list.data());
  }
  VoxelIDType* getDeviceIdPtr()
  {
    return thrust::raw_pointer_cast(m_dev_id_list.data());
  }
  Vector3ui* getDeviceCoordPtr()
  {
    return thrust::raw_pointer_cast(m_dev_coord_list.data());
  }

  inline virtual void* getVoidDeviceDataPtr()
  {
    return (void*) thrust::raw_pointer_cast(m_dev_list.data());
  }

  //! get the side length of the voxels.
  inline virtual float getVoxelSideLength() const
  {
    return m_voxel_side_length;
  }

  virtual void copyCoordsToHost(std::vector<Vector3ui>& host_vec);

  // ------ BEGIN Global API functions ------
  virtual void insertPointCloud(const std::vector<Vector3f> &points, const BitVoxelMeaning voxel_meaning);

  virtual void insertPointCloud(const PointCloud &pointcloud, const BitVoxelMeaning voxel_meaning);

  virtual void insertPointCloud(const Vector3f* d_points, uint32_t size, const BitVoxelMeaning voxel_meaning);


  virtual void insertCoordinateList(const std::vector<Vector3ui> &coordinates, const BitVoxelMeaning voxel_meaning);

  virtual void insertCoordinateList(const Vector3ui* d_coordinates, uint32_t size, const BitVoxelMeaning voxel_meaning);

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

  virtual bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud *robot_links,
                                                          const std::vector<BitVoxelMeaning>& voxel_meanings = std::vector<BitVoxelMeaning>(),
                                                          const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks = std::vector<BitVector<BIT_VECTOR_LENGTH> >(),
                                                          BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = NULL);

  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3f &metric_offset, const BitVoxelMeaning* new_meaning = NULL);
  virtual bool merge(const GpuVoxelsMapSharedPtr other, const Vector3i &voxel_offset = Vector3i(), const BitVoxelMeaning* new_meaning = NULL);

  virtual bool subtract(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3f &metric_offset = Vector3f());
  virtual bool subtract(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3i &voxel_offset = Vector3i());

  virtual bool subtractFromCountingVoxelList(const TemplateVoxelList<BitVectorVoxel, VoxelIDType> *other, const Vector3f &metric_offset = Vector3f());
  virtual bool subtractFromCountingVoxelList(const TemplateVoxelList<BitVectorVoxel, VoxelIDType> *other, const Vector3i &voxel_offset = Vector3i());

  virtual void resize(size_t new_size);

  virtual void shrinkToFit();

  virtual std::size_t getMemoryUsage() const;

  virtual void clearMap();
  //! set voxel occupancies for a specific voxelmeaning to zero

  virtual Vector3f getCenterOfMass() const;
  virtual Vector3f getCenterOfMass(Vector3ui lower_bound, Vector3ui upper_bound) const;

  virtual bool writeToDisk(const std::string path);

  virtual bool readFromDisk(const std::string path);

  virtual Vector3ui getDimensions() const;

  virtual Vector3f getMetricDimensions() const;

  // ------ END Global API functions ------

  /**
   * @brief extractCubes Extracts a cube list for visualization
   * @param [out] output_vector Resulting cube list
   */
  virtual void extractCubes(thrust::device_vector<Cube>** output_vector) const;

  /**
   * @brief collideVoxellists Internal binary search between voxellists
   * @param other Other Voxellist
   * @param offset Offset of other map to this map
   * @param collision_stencil Binary vector storing the collisions. Has to be the size of 'this'
   * @return Number of collisions
   */
  // virtual size_t collideVoxellists(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3i &offset,
  //                                  thrust::device_vector<bool>& collision_stencil) const;

  size_t collideVoxellists(const TemplateVoxelList<BitVectorVoxel, VoxelIDType> *other, const Vector3i &offset,
                                   thrust::device_vector<bool>& collision_stencil) const;

  size_t collideVoxellists(const TemplateVoxelList<CountingVoxel, VoxelIDType> *other, const Vector3i &offset,
                                   thrust::device_vector<bool>& collision_stencil) const;

  /**
   * @brief collisionCheckWithCollider
   * @param other Other VoxelList
   * @param collider Collider object
   * @param offset Offset of other map to this map
   * @return number of collisions
   */
  template< class OtherVoxel, class Collider>
  size_t collisionCheckWithCollider(const TemplateVoxelList<OtherVoxel, VoxelIDType>* other, Collider collider = DefaultCollider(), const Vector3i &offset = Vector3i());

  /**
   * @brief collisionCheckWithCollider
   * @param other Other voxelmap
   * @param collider Collider object
   * @param offset Offset of other map to this map
   * @return number of collisions
   */
  template< class OtherVoxel, class Collider>
  size_t collisionCheckWithCollider(const voxelmap::TemplateVoxelMap<OtherVoxel>* other, Collider collider = DefaultCollider(), const Vector3i &offset = Vector3i());

  /**
   * @brief equals compares two voxellists by their elements.
   * @return true, if all elements are equal, false otherwise
   */
  template< class OtherVoxel, class OtherVoxelIDType>
  bool equals(const TemplateVoxelList<OtherVoxel, OtherVoxelIDType>& other) const;

  /**
   * @brief screendump Prints ALL elemets of the list to the screen
   */
  virtual void screendump(bool with_voxel_content = true) const;

  virtual void clone(const TemplateVoxelList<Voxel, VoxelIDType>& other);

  struct VoxelToCube
  {
    VoxelToCube() {}

    __host__ __device__
    Cube operator()(const Vector3ui& coords, const BitVectorVoxel& voxel) const {

      return Cube(1, coords, voxel.bitVector());
    }
    __host__ __device__
    Cube operator()(const Vector3ui& coords, const CountingVoxel& voxel) const {

      if (voxel.getCount() > 0)
      {
        return Cube(1, coords, eBVM_OCCUPIED);
      }
      else
      {
        return Cube(1, coords, eBVM_FREE);
      }
    }
  };

  /* ======== Variables with content on device ======== */
  /* Follow the Thrust paradigm: Struct of Vectors */
  /* need to be public in order to be accessed by TemplateVoxelLists with other template arguments*/
  thrust::device_vector<VoxelIDType> m_dev_id_list;  // contains the voxel adresses / morton codes (This can not be a Voxel*, as Thrust can not sort pointers)
  thrust::device_vector<Vector3ui> m_dev_coord_list; // contains the voxel metric coordinates
  thrust::device_vector<Voxel> m_dev_list;           // contains the actual data: bitvector or probability
protected:

  virtual void remove_out_of_bounds();
  virtual void make_unique();

  /* ======== Variables with content on host ======== */
  float m_voxel_side_length;
  Vector3ui m_ref_map_dim;

  uint32_t m_blocks;
  uint32_t m_threads;

  //! result array for collision check
  bool* m_collision_check_results;
  //! result array for collision check with counter
  uint16_t* m_collision_check_results_counter;


  //! results of collision check on device
  bool* m_dev_collision_check_results;

  //! result array for collision check with counter on device
  uint16_t* m_dev_collision_check_results_counter;
};

} // end of namespace voxellist
} // end of namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H
