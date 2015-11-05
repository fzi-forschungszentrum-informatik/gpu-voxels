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

// CUDA Specific defines:
const uint32_t cMAX_NR_OF_DEVICES = 10;
const uint32_t cMAX_NR_OF_BLOCKS = 65535;
const uint32_t cMAX_THREADS_PER_BLOCK = 1024;

/*!
 * \brief BIT_VECTOR_LENGTH determines the amount of identifieable subvolumes in BitVoxels
 * The BitVoxelMeaning is stored in a Bit-Vecor of this lenghts.
 */
static const std::size_t BIT_VECTOR_LENGTH = 256;
/*!
 * the BitVoxelMeaning determines the belonging of the voxel.
 */
enum BitVoxelMeaning
{
  eBVM_OCCUPIED           = 0,
  eBVM_COLLISION          = 1,
  eBVM_FREE               = 8,
  eBVM_UNKNOWN            = 9,
  eBVM_SWEPT_VOLUME_START = 10,
  eBVM_SWEPT_VOLUME_END   = 254,
  eBVM_UNDEFINED          = 255
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

static const std::string GPU_VOXELS_MAP_TYPE_NOT_IMPLMENETED = "THIS TYPE OF DATA STRUCTURE IS NOT YET IMPLEMENTED!";
static const std::string GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED = "THIS OPERATION IS NOT SUPPORTED BY THE DATA STRUCTURE!";
static const std::string GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_0 = "THIS DATA STRUCTURE ONLY SUPPORTS BITVOXEL MEANING 0!";
static const std::string GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED = "THIS OPERATION IS NOT YET SUPPORTED BY THE DATA STRUCTURE!";
static const std::string GPU_VOXELS_MAP_SWAP_FOR_COLLIDE = "TRY TO SWAP BOTH DATA STRUCTURES TO COLLIDE.";
static const std::string GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE = "OFFSET ADDITION ONLY POSSIBLE WHEN COLLIDING WITH VOXELMAP/VOXELLIST";

// ################ Definition of the data structures build into the gpu_voxels library  #######################
// Also have a look at
//      gpu_voxels/octree/Octree.cu
//      gpu_voxels/VoxelMap.cu
// for explicit instantiation of this data structures!



// ---------------- Global in GPU Voxels ----------------------
// The following classes are used in maps, octrees and lists.
template<std::size_t length>
class BitVoxel;
class ProbabilisticVoxel;

typedef BitVoxel<BIT_VECTOR_LENGTH> BitVectorVoxel;

typedef uint32_t MapVoxelID;    // 32 Bits are enough in Maps, as GPU RAM restricts map size
typedef uint64_t OctreeVoxelID; // Doesn't have to be in a 32 bit range, since the modeled space on Octrees can be a multitude of the maps


// ---------------- VoxelMap ----------------------
namespace voxelmap {

// Forwards declaration of needed classes
class ProbVoxelMap;

template<std::size_t length>
class BitVoxelMap;

// typedefs for convenient usage
typedef BitVoxelMap<BIT_VECTOR_LENGTH> BitVectorVoxelMap;


}  // end of ns
// ------------------------------------------------

// ---------------- VoxelList ----------------------
namespace voxellist {

template<size_t length, typename VoxelIDType> class BitVoxelList;

// typedefs for convenient usage
typedef BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID> BitVectorVoxelList;
typedef BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID> BitVectorMortonVoxelList;

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

// ------------------ Visualization ----------------------
namespace visualization {
  static const size_t MAX_DRAW_TYPES = BIT_VECTOR_LENGTH;
}
// ------------------------------------------------

// ##############################################################################################################

// ##################### Block Reductions taken from Octree ##############################
// reduction in shared memory
#define REDUCE(shared,idx,nThreads,op) do { for (int r = nThreads/2; r != 0; r /= 2) {\
                                              if (idx < r) shared[idx] = shared[idx] op shared[idx + r];\
                                              __syncthreads(); } } while (0);
// reduction in shared memory
#define REDUCE2(shared1,shared2,idx,nThreads,op1,op2,branching_factor) do { for (int r = branching_factor/2; r != 0; r /= 2) {\
                                                                              if ((idx % branching_factor) < r)\
                                                                              {\
                                                                                shared1[idx] = shared1[idx] op1 shared1[idx + r];\
                                                                                shared2[idx] = shared2[idx] op2 shared2[idx + r];\
                                                                              }\
                                                                              __syncthreads(); } } while (0);

#define PARTIAL_REDUCE(shared, idx, nThreads, block_size, op) {\
  const uint32_t idx_suffix = idx & (block_size - 1); \
  for(uint32_t r = block_size / 2; r != 0; r /= 2) \
  { \
    if(idx_suffix < r) \
      shared[idx] = op(shared[idx], shared[idx + r]); \
    if(r > WARP_SIZE) \
      __syncthreads();\
  }\
}

/*
 #define PARTIAL_REDUCE(shared, idx, nThreads, size, final_size, op) {\
  for(int r = 1; r < (size / nThreads) ; ++r) \
    shared[idx] = op(shared[idx], shared[idx + r * nThreads]); \
  if((size / nThreads) > 1) \
    __syncthreads(); \
  for (int r = nThreads / 2; r >= final_size; r /= 2) \
  { \
    if(idx < r) shared[idx] = op(shared[idx], shared[idx + r]); \
    if(r > WARP_SIZE) __syncthreads();\
  } \
}
 */


    }// end of namespace
#endif
