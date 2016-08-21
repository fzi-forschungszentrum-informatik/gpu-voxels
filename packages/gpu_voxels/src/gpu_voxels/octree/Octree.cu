// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-13-21
 *
 */
//----------------------------------------------------------------------/*
#include "Octree.h"
#include "NTree.hpp"
#include <gpu_voxels/octree/GvlNTree.hpp>
#include <gpu_voxels/voxelmap/VoxelMap.hpp>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels {
namespace NTree {

// ########## Explicit Instantiation ##########
#define InstantiateNTree(branching_factor, level_count, INNER_NODE, LEAF_NODE) \
    template class NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>;\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<true,  false, false, gpu_voxels::ProbabilisticVoxel>(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::ProbabilisticVoxel>&, gpu_voxels::ProbabilisticVoxel*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<false, false, false, gpu_voxels::ProbabilisticVoxel>(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::ProbabilisticVoxel>&, gpu_voxels::ProbabilisticVoxel*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<true,  false, false, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<false, false, false, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<true,  true,  false, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_morton<true,  false, false, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxellist::TemplateVoxelList<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>, OctreeVoxelID >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, voxel_count*);\
    \
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<true,  false, true, gpu_voxels::ProbabilisticVoxel>(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::ProbabilisticVoxel>&, gpu_voxels::ProbabilisticVoxel*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<false, false, true, gpu_voxels::ProbabilisticVoxel>(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::ProbabilisticVoxel>&, gpu_voxels::ProbabilisticVoxel*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<true,  false, true, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<false, false, true, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<true,  true,  true, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxelmap::TemplateVoxelMap<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, gpu_voxels::Vector3i, voxel_count*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_morton<true,  false, true, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH> >(gpu_voxels::voxellist::TemplateVoxelList<gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>, OctreeVoxelID >&, gpu_voxels::BitVoxel<BIT_VECTOR_LENGTH>*, const uint32_t, voxel_count*);\
    \
    template OctreeVoxelID NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_load_balance<Environment::InnerNode, Environment::LeafNode, DefaultCollider>(NTree<branching_factor, level_count, Environment::InnerNode, Environment::LeafNode>*, const uint32_t, DefaultCollider, bool, double*, int*);\
    template OctreeVoxelID NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_load_balance<Environment::InnerNodeProb, Environment::LeafNodeProb, DefaultCollider>(NTree<branching_factor, level_count, Environment::InnerNodeProb, Environment::LeafNodeProb>*, const uint32_t, DefaultCollider, bool, double*, int*);

/*
// template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<1, true, false, gpu_voxels::Voxel, true>(gpu_voxels::voxelmap::VoxelMap&,BitVector<1>*, const uint32_t, gpu_voxels::Vector3ui);\
// template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_sparse<1, false, false, gpu_voxels::Voxel, true>(gpu_voxels::voxelmap::VoxelMap&,BitVector<1>*, const uint32_t, gpu_voxels::Vector3ui);\
//    template OctreeVoxelID lb_intersectNTree<branching_factor, level_count, InnerNode, LeafNode, InnerNode, LeafNode, true, DefaultCollider>(NTree<branching_factor, level_count, InnerNode, LeafNode>*, NTree<branching_factor, level_count, InnerNode, LeafNode>*, DefaultCollider, const uint32_t, double*, int*);\
//    template OctreeVoxelID lb_intersectNTree<branching_factor, level_count, InnerNode, LeafNode, InnerNode, LeafNode, false, DefaultCollider>(NTree<branching_factor, level_count, InnerNode, LeafNode>*, NTree<branching_factor, level_count, InnerNode, LeafNode>*, DefaultCollider, const uint32_t, double*, int*);

//    template OctreeVoxelID NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_load_balance<DefaultCollider>(NTree<branching_factor, level_count, InnerNode, LeafNode>*, const uint32_t, DefaultCollider, bool, double*, int*);
//    template voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_load_balance<1, false, true, gpu_voxels::RobotVoxel, true>(gpu_voxels::VoxelMap&, gpu_voxels::Vector3ui, uint32_t, BitVector<1>*);\


    template OctreeVoxelID NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_load_balance<DefaultCollider>(NTree<branching_factor, level_count, InnerNode, LeafNode>*, const uint32_t, DefaultCollider, bool, double*, int*);
    template voxel_count NTree<branching_factor, level_count, InnerNode, LeafNode>::intersect_load_balance<1, false, true, gpu_voxels::RobotVoxel, true>(gpu_voxels::VoxelMap&, gpu_voxels::Vector3ui, uint32_t, BitVector<1>*);\

    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_load_balance<1, true, false, gpu_voxels::Voxel, true>(gpu_voxels::voxelmap::VoxelMap&, gpu_voxels::Vector3ui, uint32_t, BitVector<1>*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_load_balance<1, false, false, gpu_voxels::Voxel, true>(gpu_voxels::voxelmap::VoxelMap&, gpu_voxels::Vector3ui, uint32_t, BitVector<1>*);\
    template voxel_count NTree<branching_factor, level_count, INNER_NODE, LEAF_NODE>::intersect_load_balance<1, false, true, gpu_voxels::Voxel, true>(gpu_voxels::voxelmap::VoxelMap&, gpu_voxels::Vector3ui, uint32_t, BitVector<1>*);\


*/

InstantiateNTree(BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNodeProb, Environment::LeafNodeProb)
// NTree used for tests
InstantiateNTree(BRANCHING_FACTOR, 9, Environment::InnerNodeProb, Environment::LeafNodeProb)

InstantiateNTree(BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNode, Environment::LeafNode)
// NTree used for tests
InstantiateNTree(BRANCHING_FACTOR, 9, Environment::InnerNode, Environment::LeafNode)

template class GvlNTree<BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNodeProb, Environment::LeafNodeProb>;
template class GvlNTree<BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNode, Environment::LeafNode>;

} // end of ns
} // end of ns

