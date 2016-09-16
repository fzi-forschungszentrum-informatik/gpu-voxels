// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
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
#include <gpu_voxels/voxellist/kernels/VoxelListOperations.h>
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

  typedef thrust::tuple<keyIterator, coordIterator, voxelIterator> keyCoordVoxelIteratorTriple;
  typedef thrust::zip_iterator<keyCoordVoxelIteratorTriple> keyCoordVoxelZipIterator;


public:
  TemplateVoxelList(const Vector3ui ref_map_dim, const float voxel_sidelength, const MapType map_type);

  //! Destructor
  virtual ~TemplateVoxelList();

  /* ======== getter functions ======== */

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

  // ------ BEGIN Global API functions ------
  virtual void insertPointCloud(const std::vector<Vector3f> &points, const BitVoxelMeaning voxel_meaning);

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

  virtual bool subtract(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3f &metric_offset = Vector3f());
  virtual bool subtract(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3i &voxel_offset = Vector3i());

  virtual void shrinkToFit();

  virtual std::size_t getMemoryUsage() const;

  virtual void clearMap();
  //! set voxel occupancies for a specific voxelmeaning to zero

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
  virtual size_t collideVoxellists(const TemplateVoxelList<Voxel, VoxelIDType> *other, const Vector3i &offset,
                                   thrust::device_vector<bool>& collision_stencil, bool do_locking = true) const;

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

  struct VoxelToCube
  {
    VoxelToCube() {}

    __host__ __device__
    Cube operator()(const Vector3ui& coords, const Voxel& voxel) const {
      // TODO: This is too much specialized. Create a getType() method for each voxel type
      // TODO: Pass correct voxel size
      return Cube(1, coords, voxel.bitVector());
    }
  };

protected:

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

//  thrust::device_vector<BitVoxel<length> > m_host_id_list;
//  thrust::device_vector<Voxel> m_host_list;

  /* ======== Variables with content on device ======== */
  /* Follow the Thrust paradigm: Struct of Vectors */
  thrust::device_vector<VoxelIDType> m_dev_id_list;   // contains the voxel adresses / morton codes (This may not be a Voxel*, as Thrust can not sort pointers)
  thrust::device_vector<Vector3ui> m_dev_coord_list; // contains the voxel metric coordinates
  thrust::device_vector<Voxel> m_dev_list;          // contains the actual data: bitvector or probability


  //! results of collision check on device
  bool* m_dev_collision_check_results;

  //! result array for collision check with counter on device
  uint16_t* m_dev_collision_check_results_counter;
};


template<class VoxelIDType>
struct offsetLessOperator
{
  ptrdiff_t addr_offset;
  offsetLessOperator(const Vector3ui &ref_map_dim, const Vector3i &offset)
  {
    addr_offset = voxelmap::getVoxelIndexSigned(ref_map_dim, offset);
  }

  __host__ __device__
  bool operator()(const VoxelIDType& this_voxel_addr, const VoxelIDType& other_voxel_addr) {
      return this_voxel_addr + addr_offset < other_voxel_addr;
  }
};

template<class Voxel, class VoxelIDType>
struct Merge : public thrust::binary_function<thrust::tuple<VoxelIDType, Voxel>,
    thrust::tuple<VoxelIDType, Voxel>, thrust::tuple<VoxelIDType, Voxel> >
{
  typedef thrust::tuple<VoxelIDType, Voxel> keyVoxelTuple;

  __host__ __device__
  keyVoxelTuple operator()(const keyVoxelTuple &lhs, const keyVoxelTuple &rhs) const
  {
    VoxelIDType l_key = thrust::get<0>(lhs);
    Voxel l_voxel = thrust::get<1>(lhs);

    VoxelIDType r_key = thrust::get<0>(rhs);
    Voxel r_voxel = thrust::get<1>(rhs);

    keyVoxelTuple ret = rhs;

    if (l_key == r_key)
    {
      thrust::get<1>(ret) = Voxel::reduce(l_voxel, r_voxel);
    }
    return ret;
  }
};

template<class VoxelIDType>
struct applyOffsetOperator : public thrust::unary_function<thrust::tuple<Vector3ui, VoxelIDType>,
    thrust::tuple<Vector3ui, VoxelIDType> >
{
  typedef thrust::tuple<Vector3ui, VoxelIDType> coordKeyTuple;

  ptrdiff_t addr_offset;
  Vector3i coord_offset;
  applyOffsetOperator(const Vector3ui &ref_map_dim, const Vector3i &offset)
  {
    coord_offset = offset;
    addr_offset = voxelmap::getVoxelIndexSigned(ref_map_dim, offset);
  }

  __host__ __device__
  coordKeyTuple operator()(const coordKeyTuple &input) const
  {
    coordKeyTuple ret;
    thrust::get<0>(ret) = thrust::get<0>(input) + coord_offset;
    thrust::get<1>(ret) = thrust::get<1>(input) + addr_offset;
    return ret;
  }
};

} // end of namespace voxellist
} // end of namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLIST_TEMPLATEVOXELLIST_H
