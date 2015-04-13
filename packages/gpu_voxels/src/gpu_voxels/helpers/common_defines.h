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
 * \author  Andreas Hermann
 * \date    2014-06-12
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_COMMON_DEFINES_H_INCLUDED
#define GPU_VOXELS_HELPERS_COMMON_DEFINES_H_INCLUDED

#include <string>

namespace gpu_voxels {

/*!
 * the VoxelType determines the belonging of the voxel.
 */
enum VoxelType
{
  eVT_OCCUPIED           = 0,
  eVT_COLLISION          = 1,
  eVT_FREE               = 8,
  eVT_UNKNOWN            = 9,
  eVT_SWEPT_VOLUME_START = 10,
  eVT_SWEPT_VOLUME_END   = 254,
  eVT_UNDEFINED          = 255
};

enum MapType {
  MT_BITVECTOR_VOXELMAP,         // 3D-Array of deterministic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Bitvector
  MT_BITVECTOR_VOXELLIST,        // List of     deterministic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Bitvector
  MT_BITVECTOR_OCTREE,           // Octree of   deterministic Voxels (identified by a Morton Code)                      that hold a Bitvector
  MT_BITVECTOR_MORTON_VOXELLIST, // List of     deterministic Voxels (identified by a Morton Code)                      that hold a Bitvector

  MT_PROBAB_VOXELMAP,            // 3D-Array of probabilistic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Probability
  MT_PROBAB_VOXELLIST,           // List of     probabilistic Voxels (identified by their Voxelmap-like Pointer adress) that hold a Probability
  MT_PROBAB_OCTREE,              // Octree of   probabilistic Voxels (identified by a Morton Code)                      that hold a Probability
  MT_PROBAB_MORTON_VOXELLIST     // List of     probabilistic Voxels (identified by a Morton Code)                      that hold a Probability
};

static const std::string GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED = "THIS OPERATION IS NOT SUPPORTED BY THE DATA STRUCTURE!";
static const std::string GPU_VOXELS_MAP_ONLY_SUPPORTS_VT_0 = "THIS DATA STRUCTURE ONLY SUPPORTS VOXEL TYPE 0!";
static const std::string GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED = "THIS OPERATION IS NOT YET SUPPORTED BY THE DATA STRUCTURE!";
static const std::string GPU_VOXELS_MAP_SWAP_FOR_COLLIDE = "TRY TO SWAP BOTH DATA STRUCTURES TO COLLIDE.";
static const std::string GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE = "OFFSET ADDITION ONLY POSSIBLE WHEN COLLIDING WITH VOXELMAP/VOXELLIST";

// ################ Definition of the data structures build into the gpu_voxels library  #######################
// Also have a look at
//      gpu_voxels/octree/Octree.cu
//      gpu_voxels/VoxelMap.cu
// for explicit instantiation of this data structures!

// ---------------- VoxelMap ----------------------
namespace voxelmap {
static const std::size_t BIT_VECTOR_LENGTH = 256;

// Forwards declaration of needed classes
class ProbVoxelMap;
class ProbabilisticVoxel;

template<std::size_t length>
class BitVoxelMap;

template<std::size_t length>
class BitVoxel;

// typedefs for convenient usage
typedef ProbVoxelMap VoxelMap;
typedef ProbabilisticVoxel Voxel;
typedef BitVoxelMap<BIT_VECTOR_LENGTH> BitVectorVoxelMap;
typedef BitVoxel<BIT_VECTOR_LENGTH> BitVectorVoxel;

}  // end of ns
// ------------------------------------------------


// ------------------ Octree ----------------------
namespace NTree {
static const std::size_t BRANCHING_FACTOR = 8;
static const std::size_t LEVEL_COUNT = 15;
static const std::size_t NUM_VOXEL = 439846511104UL;

// Forwards declaration of needed classes
template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
class NTree;

template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
class GvlNTree;

template<typename InnerNode, typename LeafNode>
class VisNTree;

namespace Environment
{
  class InnerNode;
  class LeafNode;
  class InnerNodeProb;
  class LeafNodeProb;
}

// Deterministic Octree
typedef NTree<BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNode, Environment::LeafNode> NTreeDet;
typedef GvlNTree<BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNode, Environment::LeafNode> GvlNTreeDet;
typedef VisNTree<Environment::InnerNode, Environment::LeafNode> VisNTreeDet;

// Probabilistic Octree
typedef NTree<BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNodeProb, Environment::LeafNodeProb> NTreeProb;
typedef GvlNTree<BRANCHING_FACTOR, LEVEL_COUNT, Environment::InnerNodeProb, Environment::LeafNodeProb> GvlNTreeProb;
typedef VisNTree<Environment::InnerNodeProb, Environment::LeafNodeProb> VisNTreeProb;

}
// ------------------------------------------------

// ##############################################################################################################

    }// end of namespace
#endif
