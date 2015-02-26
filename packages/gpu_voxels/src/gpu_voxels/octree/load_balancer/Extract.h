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
 * \date    2014-08-08
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCER_EXTRACT_H_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCER_EXTRACT_H_INCLUDED

#include <gpu_voxels/octree/load_balancer/AbstractLoadBalancer.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

typedef RunConfig<128> ExtractRunConfig; // Number of threads due to experimental founding

/**
 * @brief Extraction of stored data from \code NTree \endcode with help of the load balancing concept.
 *
 * @tparam branching_factor Branching factor of the corresponding \code NTree \endcode
 * @tparam level_count Number of levels of the corresponding \code NTree \endcode
 * @tparam InnerNode Inner node type of the corresponding \code NTree \endcode
 * @tparam LeafNode Leaf node type of the corresponding \code NTree \endcode
 * @tparam clear_collision_flag \code true \endcode to clear the collision flag while extracting data.
 * @tparam count_mode \code true \endcode to only count the number of items to extract.
 */
template<std::size_t branching_factor,
    std::size_t level_count,
    class InnerNode,
    class LeafNode,
    bool clear_collision_flag,
    bool count_mode>
class Extract : public AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItemExtract<InnerNode>, ExtractRunConfig>
{
public:
  typedef typename InnerNode::NodeData NodeData;
  typedef WorkItemExtract<InnerNode> WorkItem;
  typedef ExtractRunConfig RunConfig;
  typedef AbstractLoadBalancer<branching_factor, level_count, InnerNode, LeafNode, WorkItem, RunConfig> Base;

   /**
   * @brief Extract constructor.
   * @param ntree NTree to extract the data from.
   * @param dev_node_data Data buffer on device to write the extracted data to.
   * @param m_node_data_size The size of \code dev_node_data \endcode
   * @param dev_status_selection Device array of size \code extract_selection_size \endcode where each entry (0 or 1) defines whether data with this status should be extracted (1) or not (0).
   * @param min_level The minimal level of \code ntree \endcode to extract data from
   */
  Extract(
      NTree<branching_factor, level_count, InnerNode, LeafNode>* ntree,
      NodeData* dev_node_data,
      const uint32_t m_node_data_size,
      uint8_t* dev_status_selection,
      const uint32_t min_level = 0);

  virtual ~Extract();

protected:
  // ---------- Abstract method implementation --------------
  virtual void doWork();
  virtual void doPostCalculations();
  // --------------------------------------------------------

  virtual bool doPreparations();
  virtual void doCleanup();

protected:
  NTree<branching_factor, level_count, InnerNode, LeafNode>* m_ntree;

  NodeData* m_dev_node_data;
  const uint32_t m_node_data_size;
  const uint32_t m_min_level;
  uint8_t* m_dev_status_selection;
  uint32_t* m_dev_global_voxel_list_count;

public:
  /**
   * @brief Holds the number of extracted elements.
   */
  uint32_t m_num_elements;

};

}
}
}

#endif
