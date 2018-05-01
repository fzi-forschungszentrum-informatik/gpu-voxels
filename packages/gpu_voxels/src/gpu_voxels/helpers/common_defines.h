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

#include <boost/filesystem/path.hpp>
#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>
#include <string>

namespace gpu_voxels {

// CUDA Specific defines:
const uint32_t cMAX_NR_OF_DEVICES = 10;
//TODO: add define to allow 512^3 sized voxelmaps
//#if __CUDACC_VER_MAJOR__ > 2
//const uint32_t cMAX_NR_OF_BLOCKS = 3*65536; //512^3 / 1024 = 128K
//#else
//TODO: remove all usages of cMAX_NR_OF_BLOCKS for array dimensions, use actual runtime dimensions instead
const uint32_t cMAX_NR_OF_BLOCKS = 65535;
//#endif
const uint32_t cMAX_THREADS_PER_BLOCK = 1024;

/*!
 * \brief BIT_VECTOR_LENGTH determines the amount of identifieable subvolumes in BitVoxels
 * The BitVoxelMeaning is stored in a Bit-Vecor of this lengths.
 */
static const std::size_t BIT_VECTOR_LENGTH = 256;
/*!
 * the BitVoxelMeaning determines the belonging of the voxel.
 */
enum BitVoxelMeaning
{
  eBVM_FREE               = 0,
  eBVM_OCCUPIED           = 1,
  eBVM_COLLISION          = 2,
  eBVM_UNKNOWN            = 3,
  eBVM_SWEPT_VOLUME_START = 4,
  eBVM_SWEPT_VOLUME_END   = 254,
  // Those can be used to update probabilisitc voxels via insertPointCloud():
  eBVM_MAX_FREE_PROB      = 4,
  eBVM_UNCERTAIN_OCC_PROB = 129,
  eBVM_MAX_OCC_PROB       = 254,
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
  MT_PROBAB_MORTON_VOXELLIST,    // List of     probabilistic Voxels (identified by a Morton Code)                      that hold a Probability
  MT_COUNTING_VOXELLIST,         // Voxellist for filtering of noise data with density treshold

  MT_DISTANCE_VOXELMAP           // 3D-Array of deterministic Voxels (identified by their Voxelmap-like Pointer adress) that hold a distance and obstacle vector
};

static const std::string GPU_VOXELS_MAP_TYPE_NOT_IMPLEMENTED = "THIS TYPE OF DATA STRUCTURE IS NOT YET IMPLEMENTED!";
static const std::string GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED = "THIS OPERATION IS NOT SUPPORTED BY THE DATA STRUCTURE!";
static const std::string GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED = "THIS DATA STRUCTURE ONLY SUPPORTS BITVOXEL MEANING eBVM_OCCUPIED!";
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
class DistanceVoxel;
class CountingVoxel;

typedef BitVoxel<BIT_VECTOR_LENGTH> BitVectorVoxel;

typedef uint32_t MapVoxelID;    // 32 Bits are enough in Maps, as GPU RAM restricts map size
typedef uint64_t OctreeVoxelID; // Doesn't have to be in a 32 bit range, since the modeled space on Octrees can be a multitude of the maps

static const int32_t DISTANCE_UNINITIALISED = 0; //0 == uninitialized;

static const int32_t PBA_OBSTACLE_DISTANCE = 0;

static const int32_t PBA_UNINITIALISED_10 = 1023; // (1 << 10) - 1
static const int32_t PBA_UNINITIALISED_16 = 32767; //SHRT_MAX
static const int32_t PBA_UNINITIALISED_32 = 2147483647; //INT_MAX

static const int MANHATTAN_DISTANCE_UNINITIALIZED = 32767; //SHRT_MAX
static const int MANHATTAN_DISTANCE_START = MANHATTAN_DISTANCE_UNINITIALIZED - 1;
static const int MANHATTAN_DISTANCE_TOO_CLOSE = MANHATTAN_DISTANCE_UNINITIALIZED - 2;

//check: if PBA_UNINITIALISED_FORW_PTR != PBA_UNINITIALISED_COORD PBA phase2 needs additional checks!
//static const int32_t PBA_UNINITIALISED_COORD = PBA_UNINITIALISED_32;
static const int32_t PBA_UNINITIALISED_COORD = PBA_UNINITIALISED_10;
//static const int16_t PBA_UNINITIALISED_COORD = PBA_UNINITIALISED_16;
//static const int16_t PBA_UNINITIALISED_COORD = PBA_UNINITIALISED_16;
static const int32_t PBA_UNINITIALISED_FORW_PTR = PBA_UNINITIALISED_COORD;

//typedef int32_t pba_fw_ptr_t;
typedef int16_t pba_fw_ptr_t;

static const int32_t MAX_OBSTACLE_DISTANCE = 2147483647; //INT_MAX

static const unsigned int PBA_BLOCKSIZE = 64; //for phase 1, 2
static const unsigned int PBA_TILE_DIM = 16; //for transposeXY
static const unsigned int PBA_M3_BLOCKX = 16; //for phase 3

static const uint32_t PBA_DEFAULT_M1_BLOCK_SIZE = PBA_BLOCKSIZE;
static const uint32_t PBA_DEFAULT_M2_BLOCK_SIZE = PBA_BLOCKSIZE;
static const uint32_t PBA_DEFAULT_M3_BLOCK_SIZE = PBA_M3_BLOCKX;

enum visualizer_distance_drawmodes {
  DISTANCE_DRAW_DEFAULT,
  DISTANCE_DRAW_TWOCOLOR_GRADIENT,
  DISTANCE_DRAW_MULTICOLOR_GRADIENT,
  DISTANCE_DRAW_VORONOI_LINEAR,
  DISTANCE_DRAW_VORONOI_SCRAMBLE,
  DISTANCE_DRAW_MODE_COUNT,      // Endmarker
  DISTANCE_DRAW_PBA_INTERMEDIATE // This mode lies after endmarker to disable it (required for debugging only).
};

/**
 * @brief Type for holding occupation probability
 */
typedef int8_t Probability;
static const Probability UNKNOWN_PROBABILITY = Probability(-128);
static const Probability MIN_PROBABILITY = Probability(-127);
static const Probability MAX_PROBABILITY = Probability(127);

/* ------------------ Temporary Sensor Model ------------ */
static const Probability cSENSOR_MODEL_FREE = -10;
static const Probability cSENSOR_MODEL_OCCUPIED = 72;


// ---------------- VoxelMap ----------------------
namespace voxelmap {

// Forwards declaration of needed classes
class ProbVoxelMap;

template<std::size_t length>
class BitVoxelMap;

// typedefs for convenient usage
typedef BitVoxelMap<BIT_VECTOR_LENGTH> BitVectorVoxelMap;

class DistanceVoxelMap;

}  // end of ns
// ------------------------------------------------

// ---------------- VoxelList ----------------------
namespace voxellist {

template<size_t length, typename VoxelIDType> class BitVoxelList;

// typedefs for convenient usage
typedef BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID> BitVectorVoxelList;
typedef BitVoxelList<BIT_VECTOR_LENGTH, OctreeVoxelID> BitVectorMortonVoxelList;

class CountingVoxelList;

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


/*! Read environment variable GPU_VOXELS_MODEL_PATH
 *  \param prepend_env_path Set to false to disable and return empty path.
 *  \returns the path
 */
static inline boost::filesystem::path getGpuVoxelsPath(bool prepend_env_path)
{
  if(prepend_env_path)
  {
    char const* tmp = std::getenv("GPU_VOXELS_MODEL_PATH");
    if (tmp == NULL)
    {
      LOGGING_ERROR(
          Gpu_voxels_helpers,
          "The environment variable 'GPU_VOXELS_MODEL_PATH' could not be read. Did you set it?" << endl);
      return boost::filesystem::path("");
    }
    return boost::filesystem::path(tmp);
  }else{
    return boost::filesystem::path("");
  }
}

const float cMBYTE2BYTE = 1024.0 * 1024.0;
const float cBYTE2MBYTE = 1.0 / cMBYTE2BYTE;

// CUDA or STD min/max, depending on compiler:
#ifdef __CUDACC__
#define MIN(x,y) min(x,y)
#define MAX(x,y) max(x,y)
#else
#define MIN(x,y) std::min(x,y)
#define MAX(x,y) std::max(x,y)
#endif

}// end of namespace
#endif
