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
 * \date    2017-04-09
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/helpers_highlevel/DistanceMapConverter.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <gpu_voxels/voxel/BitVoxel.hpp>

namespace gpu_voxels
{
namespace distance_map_converter
{

using namespace voxellist;




__global__
void kernelConvertToBitVectorVoxellist(const free_space_t *free_space_list, const MapVoxelID *voxel_id_list, size_t num_elemets, const Vector3ui dims,
                                       MapVoxelID *ret_voxel_id, Vector3ui *ret_coords, BitVectorVoxel *ret_voxel)
{
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elemets;
      i += blockDim.x * gridDim.x)
  {

    // cast to larger range, to prevent overflow
    uint16_t free_space =  free_space_list[i] + eBVM_SWEPT_VOLUME_START; // Offset


    BitVectorVoxel new_voxel;
    new_voxel.bitVector().setBit(eBVM_OCCUPIED);

    // upper cap
    if(free_space > (eBVM_SWEPT_VOLUME_END + 1))
    {
      new_voxel.bitVector().setBit(eBVM_UNDEFINED);
      free_space = (eBVM_SWEPT_VOLUME_END + 1);
    }

    for(size_t sv_id = eBVM_SWEPT_VOLUME_START; sv_id < free_space;  ++sv_id)
    {
      new_voxel.bitVector().setBit(sv_id);
    }


    Vector3ui pos;
    MapVoxelID linear_id = voxel_id_list[i];
    MapVoxelID linear_id_tmp = linear_id;
    pos.z = linear_id_tmp / (dims.x * dims.y);
    pos.y = (linear_id_tmp -= pos.z * (dims.x * dims.y)) / dims.x;
    pos.x = (linear_id_tmp -= pos.y * dims.x);


    ret_coords[i] = pos;
    ret_voxel[i] = new_voxel;
    ret_voxel_id[i] = linear_id;
  }
}




/*!
 * \brief The transform_to_bitvoxel struct
 * This takes a distance voxel and generates a bitvoxel from it.
 * Therefore it calculates the distance voxels clearance and
 * maps it to the SweptVolume Bits:
 * No free space: No SV-ID set.
 * 1 Unit free space: SV-ID 1 set.
 * 2 Units free space: SV-ID 1 + 2 set.
 * ...
 * 255 Units free space: All SV-IDs set.
 * More than 255 Units free: All SV-IDs set + Undefined Bit set.
 *
 */
struct transform_to_bitvoxel : public thrust::unary_function< thrust::tuple<free_space_t, MapVoxelID >,
    thrust::tuple<MapVoxelID, Vector3ui, BitVectorVoxel> >
{
  typedef thrust::tuple<MapVoxelID, Vector3ui, BitVectorVoxel> keyCoordVoxelTriple;
  typedef thrust::tuple<free_space_t, MapVoxelID > dist_tuple_t;

  Vector3ui dims;


  __host__ __device__
  transform_to_bitvoxel(Vector3ui dims_) :
    dims(dims_) {}

  __host__ __device__
  keyCoordVoxelTriple operator()(const dist_tuple_t &tuple) const {
    keyCoordVoxelTriple ret_triple;

    // get pos from zipiterator/tuple
    uint16_t free_space = thrust::get<0>(tuple); // cast to larger range, to prevent overflow
    MapVoxelID linear_id = thrust::get<1>(tuple);

    // pos is the position of the voxel dv
    Vector3ui pos;
    MapVoxelID linear_id_tmp = linear_id;
    pos.z = linear_id_tmp / (dims.x * dims.y);
    pos.y = (linear_id_tmp -= pos.z * (dims.x * dims.y)) / dims.x;
    pos.x = (linear_id_tmp -= pos.y * dims.x);

    free_space += eBVM_SWEPT_VOLUME_START; // Offset


    BitVectorVoxel ret_voxel;
    ret_voxel.bitVector().setBit(eBVM_OCCUPIED);

    // upper cap
    if(free_space > (eBVM_SWEPT_VOLUME_END + 1))
    {
      ret_voxel.bitVector().setBit(eBVM_UNDEFINED);
      free_space = (eBVM_SWEPT_VOLUME_END + 1);
    }

    for(size_t sv_id = eBVM_SWEPT_VOLUME_START; sv_id < free_space;  ++sv_id)
    {
      ret_voxel.bitVector().setBit(sv_id);
    }

    thrust::get<0>(ret_triple) = linear_id;
    thrust::get<1>(ret_triple) = pos;
    thrust::get<2>(ret_triple) = ret_voxel;

    return ret_triple;
  }
};





size_t extract_given_distances(const voxelmap::DistanceVoxelMap &dist_map,
                               free_space_t min_dist, free_space_t max_dist,
                               BitVectorVoxelList& result) {

  // Step 1: Create an Vector containing the free-space distances of all voxels:
  thrust::device_vector<free_space_t> distances(dist_map.getVoxelMapSize());
  dist_map.extract_distances(thrust::raw_pointer_cast(distances.data()), 0);
  // Step 2: Count distances that match criteria and allocate a Bitvecor-Voxellist of that length
  size_t num_matching_voxels = thrust::count_if(distances.begin(), distances.end(), in_range(min_dist, max_dist));
  result.resize(num_matching_voxels);
  thrust::device_vector<free_space_t> matching_distances_dists(num_matching_voxels);
  thrust::device_vector<MapVoxelID> matching_distances_ids(num_matching_voxels);
  // Step 3: Copy matching Distance voxels into two new vectors
  thrust::counting_iterator<MapVoxelID> count_start(0);
  thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(distances.begin(), count_start)),
                    thrust::make_zip_iterator(thrust::make_tuple(distances.end(), count_start + dist_map.getVoxelMapSize())),
                    thrust::make_zip_iterator(thrust::make_tuple(matching_distances_dists.begin(), matching_distances_ids.begin())),
                    in_range_tuple(min_dist, max_dist));
  // Step 4: Transform distances and IDs into a Voxellist
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(matching_distances_dists.begin(), matching_distances_ids.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(matching_distances_dists.end(), matching_distances_ids.end())),
                    result.getBeginTripleZipIterator(),
                    transform_to_bitvoxel(dist_map.getDimensions()));



//  uint32_t num_blocks, threads_per_block;
//  computeLinearLoad(num_matching_voxels, &num_blocks, &threads_per_block);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  kernelConvertToBitVectorVoxellist<<<num_blocks, threads_per_block>>>(thrust::raw_pointer_cast(matching_distances_dists.data()),
//                                                                       thrust::raw_pointer_cast(matching_distances_ids.data()),
//                                                                       num_matching_voxels, dist_map.getDimensions(),
//                                                                       result.getDeviceIdPtr(), result.getDeviceCoordPtr(), result.getDeviceDataPtr());
//  CHECK_CUDA_ERROR();
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  return num_matching_voxels;


}

}
}
