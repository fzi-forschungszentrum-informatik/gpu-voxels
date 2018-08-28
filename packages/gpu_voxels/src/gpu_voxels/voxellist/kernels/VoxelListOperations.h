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
* \date    2015-05-05
*
*/
//----------------------------------------------------------------------

#ifndef GPU_VOXELS_VOXELLISTOPERATIONS_H
#define GPU_VOXELS_VOXELLISTOPERATIONS_H

#include <cuda_runtime.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/voxel/BitVoxel.h>

namespace gpu_voxels {
namespace voxellist {

// =================== VOXELMAP KERNELS ======================

/*!
 * Inserts pointcloud with global coordinates.
 * The Voxels are identified via Voxelmap-Coordinates
 */
template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const float voxel_side_length,
                                  const Vector3f* points, const std::size_t sizePoints,
                                  const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning);

/*!
 * Insert voxel coordinate tuples
 */
template<class Voxel>
__global__
void kernelInsertCoordinateTuples(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const Vector3ui* coordinates, const std::size_t sizeVoxels,
                                  const uint32_t offset_new_voxels, const BitVoxelMeaning voxel_meaning);

/**
 * @brief kernelInsertMetaPointCloud Inserts a MetaPointCloud into the voxellist
 * @param voxellist Device data-pointer of voxellist
 * @param voxel_side_length Metapointcloud only contains vertices. Choose a sidelength of inserted voxels
 * @param meta_point_cloud This metapointcloud should be inserted
 * @param offset_new_points Offset which is added to the metapointcloud coordinates
 * @param voxel_meaning Defines the set bits in the generated BitVoxels
 */
template<class Voxel>
__global__
void kernelInsertMetaPointCloud(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui ref_map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning);

/**
 * @brief kernelInsertMetaPointCloud Inserts a MetaPointCloud into the voxellist
 * @param voxellist Device data-pointer of voxellist
 * @param voxel_side_length Metapointcloud only contains vertices. Choose a sidelength of inserted voxels
 * @param meta_point_cloud This metapointcloud should be inserted
 * @param offset_new_points Offset which is added to the metapointcloud coordinates
 * @param voxel_meanings Array of voxelmeanings. Each cloud in the Metapointcloud has its own point meaning
 */
template<class Voxel>
__global__
void kernelInsertMetaPointCloud(MapVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning* voxel_meanings);



/**
 * @brief kernelCollideWithVoxelMap Collision check kernel between a voxellist and a voxelmap
 * @param [in] this_list Device pointer to this list
 * @param [in] this_list_size Number of voxels in this list
 * @param [in] other_map Device pointer to other map
 * @param [in] other_map_size Number of voxels in other map
 * @param [in] offset The other map can be offset for collision check
 * @param [out] results array of number of collisions (one entry for each block)
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__

void kernelCollideWithVoxelMap(const MapVoxelID* this_id_list, Voxel* this_voxel_list, uint32_t this_list_size,
                               const OtherVoxel* other_map, Vector3ui other_map_dim, Collider collider,
                               Vector3i offset, uint16_t* results);


/**
 * @brief kernelCollideWithVoxelMap Collision check kernel between a voxellist and a probabilistic voxelmap
 * returning the bits in collision
 * @param [in] this_id_list Device pointer to this lists IDs
 * @param [in] this_voxel_list Device pointer to this lists Bitvoxels
 * @param [in] this_list_size Number of voxels in this list
 * @param [in] other_map Device pointer to other map
 * @param [in] other_map_dim Number of voxels in other map
 * @param [in] col_threshold When to inspect a occupied Voxel
 * @param [in] offset The other map can be offset for collision check
 * @param [out] results array of Bitvectors of collisions (one entry for each block)
 */
/*__global__
void kernelCollideWithVoxelMap(const MapVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const ProbabilisticVoxel* other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* results);
*/

template<class VoxelType>
__global__
void kernelCollideWithVoxelMap(const MapVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results);

template<class VoxelType>
__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* bitvoxel_results);

/**
 * @brief kernelCollideWithVoxelMapBitMask Collision check kernel between a voxellist and a probabilistic voxelmap
 * that only detects and counts collisions for specific BitVoxelMeanings
 * @param [in] this_id_list Device pointer to this lists IDs
 * @param [in] this_voxel_list Device pointer to this lists Bitvoxels
 * @param [in] this_list_size Number of voxels in this list
 * @param [in] other_map Device pointer to other map
 * @param [in] other_map_dim Number of voxels in other map
 * @param [in] col_threshold When to inspect a occupied Voxel
 * @param [in] offset The other map can be offset for collision check
 * @param [in] bitvoxel_mask The mask that specifies, which BVMs to check
 * @param [out] coll_counter_results Array of Bitvectors of collisions (one entry for each block)
 */
/*__global__
void kernelCollideWithVoxelMapBitMask(const MapVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const ProbabilisticVoxel *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results);
*/
template<class VoxelType>
__global__
void kernelCollideWithVoxelMapBitMask(const MapVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results);

template<class VoxelType>
__global__
void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const VoxelType *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results);

// =================== MORTON KERNELS ======================

/*!
 * Inserts pointcloud with global coordinates.
 * The Voxels are identified via Morton-Coordinates
 */
template<class Voxel>
__global__
void kernelInsertGlobalPointCloud(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const float voxel_side_length,
                                  const Vector3f* points, const std::size_t sizePoints,
                                  const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning);
template<class Voxel>
__global__
void kernelInsertCoordinateTuples(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                  const Vector3ui ref_map_dim, const Vector3ui* coordinates, const std::size_t sizePoints,
                                  const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning);

/**
 * @brief kernelInsertMetaPointCloud Inserts a MetaPointCloud into the Morton-Voxellist
 * @param voxellist Device data-pointer of voxellist
 * @param voxel_side_length Metapointcloud only contains vertices. Choose a sidelength of inserted voxels
 * @param meta_point_cloud This metapointcloud should be inserted
 * @param offset_new_points Offset which is added to the metapointcloud coordinates
 * @param voxel_meaning Defines the set bits in the generated BitVoxels
 */
template<class Voxel>
__global__
void kernelInsertMetaPointCloud(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui ref_map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning voxel_meaning);

/**
 * @brief kernelInsertMetaPointCloud Inserts a MetaPointCloud into the Morton-Voxellist
 * @param voxellist Device data-pointer of voxellist
 * @param voxel_side_length Metapointcloud only contains vertices. Choose a sidelength of inserted voxels
 * @param meta_point_cloud This metapointcloud should be inserted
 * @param offset_new_points Offset which is added to the metapointcloud coordinates
 * @param voxel_meanings Array of voxelmeanings. Each cloud in the Metapointcloud has its own point meaning
 */
template<class Voxel>
__global__
void kernelInsertMetaPointCloud(OctreeVoxelID* id_list, Vector3ui* coord_list, Voxel* voxel_list,
                                const Vector3ui map_dim, const float voxel_side_length,
                                const MetaPointCloudStruct *meta_point_cloud,
                                const uint32_t offset_new_points, const BitVoxelMeaning* voxel_meanings);





/**
 * @brief kernelCollideWithVoxelMap Collision check kernel between a voxellist and a voxelmap
 * @param [in] this_list Device pointer to this list
 * @param [in] this_list_size Number of voxels in this list
 * @param [in] other_map Device pointer to other map
 * @param [in] other_map_size Number of voxels in other map
 * @param [in] offset The other map can be offset for collision check
 * @param [out] results array of number of collisions (one entry for each block)
 */
template<class Voxel, class OtherVoxel, class Collider>
__global__

void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, Voxel* this_voxel_list, uint32_t this_list_size,
                               const OtherVoxel* other_map, Vector3ui other_map_dim, Collider collider,
                               Vector3i offset, uint16_t* results);


/**
 * @brief kernelCollideWithVoxelMap Collision check kernel between a voxellist and a probabilistic voxelmap
 * returning the bits in collision
 * @param [in] this_id_list Device pointer to this lists IDs
 * @param [in] this_voxel_list Device pointer to this lists Bitvoxels
 * @param [in] this_list_size Number of voxels in this list
 * @param [in] other_map Device pointer to other map
 * @param [in] other_map_dim Number of voxels in other map
 * @param [in] col_threshold When to inspect a occupied Voxel
 * @param [in] offset The other map can be offset for collision check
 * @param [out] results array of Bitvectors of collisions (one entry for each block)
 */
__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const ProbabilisticVoxel* other_map, Vector3ui other_map_dim, float col_threshold,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* results);

__global__
void kernelCollideWithVoxelMap(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                               const BitVectorVoxel* other_map, Vector3ui other_map_dim,
                               Vector3i offset, uint16_t* coll_counter_results, BitVectorVoxel* results);

/**
 * @brief kernelCollideWithVoxelMapBitMask Collision check kernel between a voxellist and a probabilistic voxelmap
 * that only detects and counts collisions for specific BitVoxelMeanings
 * @param [in] this_id_list Device pointer to this lists IDs
 * @param [in] this_voxel_list Device pointer to this lists Bitvoxels
 * @param [in] this_list_size Number of voxels in this list
 * @param [in] other_map Device pointer to other map
 * @param [in] other_map_dim Number of voxels in other map
 * @param [in] col_threshold When to inspect a occupied Voxel
 * @param [in] offset The other map can be offset for collision check
 * @param [in] bitvoxel_mask The mask that specifies, which BVMs to check
 * @param [out] coll_counter_results Array of Bitvectors of collisions (one entry for each block)
 */
__global__
void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const ProbabilisticVoxel *other_map, Vector3ui other_map_dim, float col_threshold,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results);

__global__
void kernelCollideWithVoxelMapBitMask(const OctreeVoxelID* this_id_list, BitVectorVoxel *this_voxel_list, uint32_t this_list_size,
                                      const BitVectorVoxel *other_map, Vector3ui other_map_dim,
                                      Vector3i offset, const BitVectorVoxel* bitvoxel_mask, uint16_t* coll_counter_results);

// =================== UNUSED KERNELS ======================
/**
 * @brief kernelCopyFromBitVoxelMap Fills voxellist from a given voxelmap
 * @param [in] dev_voxels Device pointer of voxelmap
 * @param [in] map_size Size of voxelmap
 * @param [in] map_dimension 3D-dimension of voxelmap
 * @param [out] output_array Device pointer to this_list voxels
 * @param [out] entries_per_block Basically this is a buffer variable to store the number of voxels per cuda block
 * @param [out] new_entries_total Number of new voxels
 * @param [in] map_offset The voxelmap will be offset by this before copying
 * @param overwrite_meaning [in] Voxels in voxellists have this meaning. Defaults to eBVM_OCCUPIED
 */
//template<class Voxel>
//__global__
//void kernelCopyFromBitVoxelMap(BitVectorVoxel* dev_voxels,
//                               const uint32_t map_size,
//                               const Vector3ui map_dimension,
//                               Voxel* output_array,
//                               size_t* entries_per_block,
//                               uint32_t* new_entries_total,
//                               const Vector3i map_offset,
//                               BitVoxelMeaning overwrite_meaning = eBVM_OCCUPIED);

} // end of namespace voxellist
} // end of namespace gpu_voxels

#endif // GPU_VOXELS_VOXELLISTOPERATIONS_H
