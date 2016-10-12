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
 * \date    2014-12-10
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_NTREE_H_INCLUDED
#define GPU_VOXELS_OCTREE_NTREE_H_INCLUDED

#include <limits.h>

// thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// internal includes
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/Morton.h>
#include <gpu_voxels/octree/NTreeData.h>
#include <gpu_voxels/octree/EnvironmentNodes.h>
#include <gpu_voxels/octree/EnvNodesProbabilistic.h>
#include <gpu_voxels/octree/DefaultCollider.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/BitVector.h>
#include <gpu_voxels/voxelmap/VoxelMap.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxellist/TemplateVoxelList.h>
#include <gpu_voxels/vis_interface/VisualizerInterface.h>

/**
 * @namespace gpu_voxels::NTree
 * Contains implementation of NTree Datastructure and according operations
 */
namespace gpu_voxels {
namespace NTree {

template<typename LeafNode>
struct FindResult
  {
  public:
    void* m_device_node_pointer;
    uint8_t m_level;
    LeafNode m_node_data; //InnerNodes are converted to LeafNodes

    __host__ __device__
    FindResult()
    {

    }

    __host__ __device__
    FindResult(void* device_node_pointer, uint8_t level, LeafNode node_data)
    {
      m_device_node_pointer = device_node_pointer;
      m_level = level;
      m_node_data = node_data;
    }
  };

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
class NTree
{
protected:
  static const uint32_t INITIAL_REBUILD_BUFFER_SIZE = 4000000;
  static const uint32_t INITIAL_EXTRACT_BUFFER_SIZE = 100000;
  //static const std::size_t MAX_MEM_USAGE = 512 * std::size_t(1024 * 1024);
  uint32_t m_extract_buffer_size;
  uint32_t m_rebuild_buffer_size;

public:
  typedef InnerNode InnerNode_Type;
  typedef LeafNode LeafNode_Type;
  typedef typename InnerNode::NodeData NodeData;
  typedef typename NodeData::BasicData BasicData;
  //const std::size_t _branching_factor = branching_factor;

  InnerNode* m_root;
//  gpu_voxels::Vector3ui origin;
//  uint32_t sideLengthInVoxel;
//  float voxelSideLength;
  thrust::device_vector<void*> m_allocation_list;
  uint32_t numBlocks, numThreadsPerBlock;
  voxel_count allocInnerNodes, allocLeafNodes;
  uint8_t* m_status_mapping;
  uint8_t* m_extract_status_selection;
  std::size_t m_max_memory_usage;
  uint32_t m_rebuild_counter;

  /**
   * Voxel side length measured in mm
   */
  uint32_t m_resolution;

  /**
   * Coordinates of the center of the octree
   */
  Vector3ui m_center;

  // Whether the holds any data. Needed to know whether to use a build() or insert() for the first point data to add to the tree
  bool m_has_data;

  NTree(uint32_t numBlocks, uint32_t numThreadsPerBlock, uint32_t resolution = 10);

  virtual ~NTree();

  /**
   * @brief Builds an NTree out of the given set of occupied points.
   * @param h_points Set of occupied points. Coordinates in meter.
   * @param free_bounding_box True to set the bounding-box of the given set of points as free space.
   */
  void build(thrust::host_vector<Vector3ui>& h_points, const bool free_bounding_box = false);

  /**
   * @brief Builds an NTree out of the given set of occupied points.
   * @param h_points Set of occupied points. Coordinates in voxels of the NTree.
   * @param free_bounding_box True to set the bounding-box of the given set of points as free space.
   */
  void build(thrust::device_vector<Vector3ui>& d_points, const bool free_bounding_box = false);

  void print();
  void print2();
  void find(thrust::device_vector<Vector3ui> voxel, void** resultNode,
            thrust::device_vector<enum NodeType> resultNodeType);

  /**
   * @brief Searches for the given Voxel \c h_voxel in the NTree and returns the found Nodes \c resultNode
   * @param h_voxel
   * @param resultNode
   */
  void find(thrust::host_vector<Vector3ui>& h_voxel, thrust::host_vector<FindResult<LeafNode> >& resultNode);

  voxel_count intersect(thrust::host_vector<Vector3ui>& h_voxel);

  /**
   * @brief intersect_sparse Intersect NTree and VoxelMap by checking every occupied voxel of the VoxelMap in the NTree. Performs good for a sparsely occupied VoxelMap.
   * @tparam set_collision_flag Boolean whether to compute set the collision flag of the NTree
   * @tparam compute_voxelTypeFlags Boolean whether to compute the intersecting BitVector
   * @tparam compute_collsWithUnknown Boolean whether to process intersections with unknown cells as collisions
   * @tparam VoxelType The type of the voxel of the given voxelmap.
   * @tparam use_execution_context Use the execution context of voxel_map.
   * @param voxel_map VoxelMap to collide with
   * @param h_result_voxel The resulting reduced voxel if compute_voxelTypeFlags=true
   * @param min_level Min level used for traversal
   * @param offset Offset that gets added to the voxelmap coordinates
   * @param num_colls_with_unknown_cells Returns the number of collisions with 'unknown' cells. Requires \code compute_collsWithUnknown \endcode to be true.
   * @return Returns the number of collisions
   */
  template<bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
  voxel_count intersect_sparse(gpu_voxels::voxelmap::TemplateVoxelMap<VoxelType>& voxel_map,
                               VoxelType* h_result_voxel = NULL, const uint32_t min_level = 0,
                               gpu_voxels::Vector3i offset = gpu_voxels::Vector3i(0, 0, 0), voxel_count* num_colls_with_unknown_cells = NULL);

  /**
   * @brief intersect_sparse Intersect NTree and VoxelList by checking every voxel of the VoxelList in the NTree.
   * @tparam set_collision_flag Boolean whether to compute set the collision flag of the NTree
   * @tparam compute_voxelTypeFlags Boolean whether to compute the intersecting BitVector
   * @tparam compute_collsWithUnknown Boolean whether to process intersections with unknown cells as collisions
   * @tparam VoxelType The type of the voxel of the given voxellist.
   * @tparam use_execution_context Use the execution context of voxel_map.
   * @param voxel_map voxellist to collide with
   * @param h_result_voxel The resulting reduced voxel if compute_voxelTypeFlags=true
   * @param min_level Min level used for traversal
   * @param offset Offset that gets added to the voxellist coordinates
   * @param num_colls_with_unknown_cells Returns the number of collisions with 'unknown' cells. Requires \code compute_collsWithUnknown \endcode to be true.
   * @return Returns the number of collisions
   */
  template<bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
  voxel_count intersect_sparse(gpu_voxels::voxellist::TemplateVoxelList<VoxelType, MapVoxelID>& voxel_map,
                               BitVectorVoxel* h_result_voxel = NULL, const uint32_t min_level = 0,
                               gpu_voxels::Vector3i offset = gpu_voxels::Vector3i(0, 0, 0), voxel_count* num_colls_with_unknown_cells = NULL);

  /**
   * @brief intersect_morton Intersect NTree and MortonVoxelList by checking every voxel of the MortonVoxelList in the NTree.
   * @tparam set_collision_flag Boolean whether to compute set the collision flag of the NTree
   * @tparam compute_voxelTypeFlags Boolean whether to compute the intersecting BitVector
   * @tparam compute_collsWithUnknown Boolean whether to process intersections with unknown cells as collisions
   * @tparam VoxelType The type of the voxel of the given voxellist.
   * @tparam use_execution_context Use the execution context of voxel_map.
   * @param voxel_map voxellist to collide with
   * @param h_result_voxel The resulting reduced voxel if compute_voxelTypeFlags=true
   * @param min_level Min level used for traversal
   * @param offset Offset that gets added to the voxellist coordinates
   * @param num_colls_with_unknown_cells Returns the number of collisions with 'unknown' cells. Requires \code compute_collsWithUnknown \endcode to be true.
   * @return Returns the number of collisions
   */
  template<bool set_collision_flag, bool compute_voxelTypeFlags, bool compute_collsWithUnknown, typename VoxelType>
  voxel_count intersect_morton(gpu_voxels::voxellist::TemplateVoxelList<VoxelType, OctreeVoxelID>& voxel_list,
                               BitVectorVoxel* h_result_voxel = NULL, const uint32_t min_level = 0, voxel_count* num_colls_with_unknown_cells = NULL);

  /**
   * @brief intersect_load_balance Intersect NTree and VoxelMap by traversing the NTree with load balance and look-up occupied voxel in the VoxelMap.
   * @tparam VTF_SIZE Size parameter for BitVector<VTF_SIZE>
   * @tparam set_collision_flag Boolean whether to compute set the collision flag of the NTree
   * @tparam compute_voxelTypeFlags Boolean whether to compute the intersecting BitVector
   * @tparam VoxelType The type of the voxel. For RobotVoxel the 'voxelType' is used and for Voxel 'occupancy_planning'/ 'occupancy_execution'
   * @tparam use_execution_context Use the execution context of voxel_map.
   * @param voxel_map VoxelMap to collide with
   * @param offset Used to translate from VoxelMap to NTree coordinates
   * @param min_level Stop at this level of the NTree with the collision check
   * @param h_result_voxelTypeFlags Colliding voxel types if use_voxelTypeFlags set to true
   * @return Returns the number of collisions
   */
  template<int vft_size, bool set_collision_flag, bool compute_voxelTypeFlags, typename VoxelType>
  voxel_count intersect_load_balance(gpu_voxels::voxelmap::ProbVoxelMap& voxel_map,
                                     gpu_voxels::Vector3i offset = gpu_voxels::Vector3i(0, 0, 0),
                                     const uint32_t min_level = 0,
                                     BitVector<vft_size>* h_result_voxelTypeFlags = NULL);

  template<typename o_InnerNode, typename o_LeafNode>
  voxel_count intersect(NTree<branching_factor, level_count, o_InnerNode, o_LeafNode>* other);


  /**
   * \brief intersect_load_balance Intersect two NTree by simultaneous depth-first traversal with load balancing.
   * @param other The NTree to collide with.
   * @param min_level Stop traversal at this tree level. Adjusts the accuracy of collision checks.
   * @param collider Collider which defines when a collisions occurs.
   * @param set_collision_flag Sets the collision flag in case of collision for any of the two nodes which are in conflict.
   * @param balance_overhead Returns the overhead for the load balancing in ms.
   * @param num_balance_tasks Returns the number of load balancing steps.
   * @return Returns the number of collisions. This number represents the volume in collision by the number of voxels (voxel with the size of the NTree resolution).
   */
  template<typename o_InnerNode, typename o_LeafNode, typename Collider>
  OctreeVoxelID intersect_load_balance(NTree<branching_factor, level_count, o_InnerNode, o_LeafNode>* other,
                                     const uint32_t min_level, Collider collider = DefaultCollider(), bool mark_collisions = true,
                                     double* balance_overhead = NULL, int* num_balance_tasks = NULL);

  /*
   * Inserts the given voxel in the tree and updates their occupancy. The given voxel have to be sorted by their id.
   */
  void insertVoxel(thrust::device_vector<Voxel>& d_free_space_voxel,
                   thrust::device_vector<Voxel>& d_object_voxel, gpu_voxels::Vector3ui sensor_origin,
                   const uint32_t free_space_resolution, const uint32_t object_resolution);

  /*
   * Inserts the given voxel in the tree and updates their occupancy. The given voxel have to be sorted by their id.
   */
  void insertVoxel(thrust::device_vector<Voxel>& d_voxel_vector, bool set_free, bool propagate_up);

  void propagate_bottom_up(thrust::device_vector<Voxel>& d_voxel_vector, uint32_t level = 0);

  void propagate_bottom_up(OctreeVoxelID* d_voxel_id, voxel_count num_voxel, uint32_t level = 0);

  /**
   * Restores the tree invariant by propagating the status flags top-down and bottom-up using the load balancing concept.
   */
  void propagate(const uint32_t num_changed_nodes = 0);

  /**
   * Sequential check of the tree whether the tree invariant is valid or not.
   * Returns true for valid.
   */
  bool checkTree();

  uint32_t extractCubes(std::vector<Vector3f> &points, uint8_t* d_status_selection = NULL, uint32_t min_level = 0);

  uint32_t extractCubes(thrust::device_vector<Cube> *&d_cubes, uint8_t* d_status_selection = NULL,
                        uint32_t min_level = 0);

  /**
   * Returns true if a rebuild operation is suggested.
   */
  bool needsRebuild() const;

  std::size_t getMemUsage() const;

  /**
   * Copy data of NTree into new one. Used for memory cleanup.
   * Returns a pointer to the new NTree.
   */
  void rebuild();

  std::size_t getMaxMemoryUsage() const
  {
    return m_max_memory_usage;
  }

  void setMaxMemoryUsage(std::size_t max_mem_usage)
  {
    m_max_memory_usage = max_mem_usage;
  }

  void clearCollisionFlags();

  void serialize(std::ostream& out, const bool bin_mode = true);

  bool deserialize(std::istream& in, const bool bin_mode = true);

  void clear();

  /*
   * ############# Some templated helpers #############
   */

  struct Trafo_NodeData_to_OctreeVoxelID
  {
    __host__ __device__ OctreeVoxelID operator()(NodeData x)
    {
      return x.m_voxel_id;
    }
  };

//  struct Trafo_NodeData_to_Cube
//  {
//    __host__ __device__
//    Cube operator()(NodeData x)
//    {
//      Cube c;
//      inv_morton_code60(x.m_voxel_id, c.m_position);
//      c.m_side_length = getVoxelSideLength<branching_factor>(x.m_level);
//      NodeStatus tmp = x.m_status;
//      if (x.m_level == 0)
//      {
//        if ((x.m_status & ns_UNKNOWN) == ns_UNKNOWN)
//          tmp = ns_UNKNOWN;
//        else if (x.m_occupancy >= THRESHOLD_OCCUPANCY)
//          tmp = ns_OCCUPIED;
//        else
//          tmp = ns_FREE;
//      }
//      c.m_type = statusToBitVoxelMeaning(tmp);
//      return c;
//    }
//  };

  struct Comp_has_level
  {
    voxel_count m_level;

    __host__ __device__ Comp_has_level(voxel_count level)
    {
      m_level = level;
    }

    __host__ __device__
    bool operator()(NodeData x)
    {
      return x.m_level == m_level;
    }
  };

  struct Trafo_to_BasicData
  {
    __host__ __device__ typename NodeData::BasicData operator()(NodeData x)
    {
      return x.m_basic_data;
    }
  };

//  struct Trafo_NodeData_to_Pair
//  {
//    __host__ __device__
//    thrust::pair<NodeStatus, Probability> operator()(NodeData x)
//    {
//      return thrust::make_pair<NodeStatus, Probability>(x.m_status, x.m_occupancy);
//    }
//  };

  struct Trafo_Pair_to_Probability
  {
    __host__ __device__ Probability operator()(thrust::pair<NodeStatus, Probability> x)
    {
      return x.second;
    }
  };

  struct Trafo_Pair_to_Status
  {
    __host__ __device__ NodeStatus operator()(thrust::pair<NodeStatus, Probability> x)
    {
      return x.first;
    }
  };

  struct Trafo_Scale_Coordinate
  {
    uint32_t m_scale;

    __host__ __device__ Trafo_Scale_Coordinate(uint32_t scale)
    {
      m_scale = scale;
    }

    __host__ __device__ uint32_t operator()(uint32_t x)
    {
      return x / m_scale;
    }
  };

  struct Trafo_OctreeVoxelID
  {
    uint32_t m_scale;

    __host__ __device__ Trafo_OctreeVoxelID(uint32_t scale)
    {
      m_scale = scale;
    }

    __host__ __device__ OctreeVoxelID operator()(OctreeVoxelID x)
    {
      gpu_voxels::Vector3ui coorinates;
      inv_morton_code60(x, coorinates);
      return morton_code60(coorinates.x * m_scale, coorinates.y * m_scale, coorinates.z * m_scale);
    }
  };

  struct Comp_is_collision
  {
    __host__ __device__
    bool operator()(const Cube x)
    {
      return x.m_type_vector.getBit(gpu_voxels::eBVM_COLLISION);
    }
  };

  struct ComputeFreeSpaceData
  {
    __host__ __device__
    ComputeFreeSpaceData()
    {
    }

    __host__ __device__
    ComputeFreeSpaceData(OctreeVoxelID* voxel_id, BasicData* basic_data, voxel_count count) :
        m_voxel_id(voxel_id), m_basic_data(basic_data), m_count(count)
    {
    }

    OctreeVoxelID* m_voxel_id;
    BasicData* m_basic_data;
    voxel_count m_count;
  };

  struct Trafo_NodeData_to_Cube
  {
    uint8_t* m_mapping_lookup;

    __host__ __device__
    Trafo_NodeData_to_Cube(uint8_t* mapping_lookup)
    {
      m_mapping_lookup = mapping_lookup;
    }

    __host__ __device__ __forceinline__
    Cube nodeDataToCube(Environment::NodeData& x)
    {
      Cube c;
      inv_morton_code60(x.m_voxel_id, c.m_position);
      c.m_position = c.m_position - Vector3ui(VISUALIZER_SHIFT_X, VISUALIZER_SHIFT_Y, VISUALIZER_SHIFT_Z);
      c.m_side_length = getVoxelSideLength<branching_factor>(x.m_level);
      c.m_type_vector.setBit(statusToBitVoxelMeaning(m_mapping_lookup, x.m_basic_data.m_status));
      //      if (c.m_type != gpu_voxels::eBVM_OCCUPIED && c.m_type != gpu_voxels::BVM_SWEPT_VOLUME_END && c.m_type != gpu_voxels::BVM_UNDEFINED)
      //        printf("Type %u Status %u\n", c.m_type, x.m_basic_data.m_status);
      return c;
    }

    __host__ __device__ __forceinline__
    Cube nodeDataToCube(Environment::NodeDataProb& x)
    {
      Cube c;
      inv_morton_code60(x.m_voxel_id, c.m_position);
      c.m_position = c.m_position - Vector3ui(VISUALIZER_SHIFT_X, VISUALIZER_SHIFT_Y, VISUALIZER_SHIFT_Z);
      c.m_side_length = getVoxelSideLength<branching_factor>(x.m_level);
      NodeStatus status = x.m_basic_data.m_status & ns_COLLISION;
      if (x.m_basic_data.m_occupancy == UNKNOWN_PROBABILITY)
        status |= ns_UNKNOWN;
      else if (x.m_basic_data.m_occupancy >= THRESHOLD_OCCUPANCY)
        status |= ns_OCCUPIED;
      else if (x.m_basic_data.m_occupancy < THRESHOLD_OCCUPANCY)
        status |= ns_FREE;
      c.m_type_vector.setBit(statusToBitVoxelMeaning(m_mapping_lookup, status));
      //      if (c.m_type != gpu_voxels::eBVM_OCCUPIED && c.m_type != gpu_voxels::BVM_SWEPT_VOLUME_END && c.m_type != gpu_voxels::BVM_UNDEFINED)
      //        printf("Type %u Status %u\n", c.m_type, x.m_basic_data.m_status);
      return c;
    }

    template<typename NodeData>
    __host__ __device__
    Cube operator()(NodeData& x)
    {
      return nodeDataToCube(x);
    }
  };

protected:

  /*
   * Method needed for inserting new sensor data. Computes the free space by ray casting and inserts it into the tree.
   */
  void computeFreeSpaceViaRayCast_VoxelList(
      thrust::device_vector<Voxel>& d_occupied_voxel, gpu_voxels::Vector3ui sensor_origin,
      thrust::host_vector<thrust::pair<OctreeVoxelID*, voxel_count> >& h_packed_levels);

  /*
   * Method needed for inserting new sensor data. Computes the free space by ray casting and inserts it into the tree.
   */
  void computeFreeSpaceViaRayCast(thrust::device_vector<Voxel>& d_occupied_voxel,
                                  gpu_voxels::Vector3ui sensor_origin,
                                  thrust::host_vector<ComputeFreeSpaceData>& h_packed_levels,
                                  uint32_t min_level = 0);

  /*
   * Inserts the given voxel in the tree and updates their occupancy. The given voxel have to be sorted by their id.
   */
  template<bool SET_UPDATE_FLAG, typename BasicData, typename Iterator1, typename Iterator2>
  void insertVoxel(OctreeVoxelID* d_voxel_vector, Iterator1 d_set_basic_data, Iterator2 d_reset_basic_data,
                   voxel_count num_voxel, uint32_t target_level);

  void init_const_memory();

  void packVoxel_Map_and_List(
      MapProperties<typename InnerNode::RayCastType, branching_factor>& map_properties,
      thrust::host_vector<thrust::pair<OctreeVoxelID*, voxel_count> >& h_packed_levels, voxel_count num_free_voxel,
      uint32_t min_level);

  void packVoxel_Map(MapProperties<typename InnerNode::RayCastType, branching_factor>& map_properties,
                     thrust::host_vector<ComputeFreeSpaceData>& h_packed_levels, voxel_count num_free_voxel,
                     uint32_t min_level);

  void free_bounding_box(thrust::device_vector<Vector3ui>& d_points);

  void toVoxelCoordinates(thrust::host_vector<Vector3f>& h_points, thrust::device_vector<Vector3ui>& d_voxels);

  void internal_rebuild(thrust::device_vector<NodeData>& d_node_data, const uint32_t num_cubes);
};

#ifndef NTREE_PRECOMPILE
#include "NTree.hpp"
#endif

} // end of ns
} // end of ns

#endif /* NTREE_H_ */
