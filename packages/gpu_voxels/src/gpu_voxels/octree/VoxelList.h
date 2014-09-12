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
 * \date    2014-01-27
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_VOXELLIST_H_INCLUDED
#define GPU_VOXELS_VOXELLIST_H_INCLUDED

#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/Morton.h>

// CUDA samples
#include <gpu_voxels/helpers/cuda_handling.h>

#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/merge.h>

#include <gpu_voxels/helpers/cuda_datatypes.h>
//#include <gpu_voxels/octree/Voxel.h>
#include <gpu_voxels/octree/DataTypes.h>

#include <gpu_voxels/logging/logging_octree.h>

#include <stdlib.h>


namespace gpu_voxels {
namespace NTree {

/*
 * Data structure which manages a sorted list of occupied voxels represented by their morton code in the gpu memory for fast intersection queries.
 */
template<int VTF_SIZE>
class VoxelList
{
private:
  thrust::device_vector<VoxelID> m_voxel_id_array;
  thrust::device_vector<VoxelTypeFlags<VTF_SIZE> > m_voxel_type_flags;

public:
  VoxelList()
  {

  }

  ~VoxelList()
  {

  }

  void insertVoxels(thrust::host_vector<gpu_voxels::Vector3ui>& voxels,
                    thrust::host_vector<VoxelTypeFlags<VTF_SIZE> >& voxelTypeFlags)
  {
    // copy to gpu
    timespec time = getCPUTime();
    thrust::device_vector<gpu_voxels::Vector3ui> d_new_voxels = voxels;
    thrust::device_vector<VoxelID> d_new_voxel_ids(voxels.size());
    thrust::device_vector<VoxelTypeFlags<VTF_SIZE> > d_voxelTypeFlags = voxelTypeFlags;
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    LOGGING_INFO(OctreeLog, "copy to gpu: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);

    // transform to morton code
    time = getCPUTime();
    transform_to_morton<gpu_voxels::Vector3ui> t;
    thrust::transform(d_new_voxels.begin(), d_new_voxels.end(), d_new_voxel_ids.begin(), t);
    d_new_voxels.clear();
    d_new_voxels.shrink_to_fit();
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    LOGGING_INFO(OctreeLog,"transformation to morton code: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);

    // sort
    time = getCPUTime();
    thrust::sort_by_key(d_new_voxel_ids.begin(), d_new_voxel_ids.end(), d_voxelTypeFlags.begin());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    LOGGING_INFO(OctreeLog, "sorting: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);

    // alloc new memory
    time = getCPUTime();
    thrust::device_vector<VoxelID> new_voxel_id_array(m_voxel_id_array.size() + voxels.size());
    thrust::device_vector<VoxelTypeFlags<VTF_SIZE> > new_voxelTypeFlags(
        m_voxel_type_flags.size() + voxelTypeFlags.size());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    LOGGING_INFO(OctreeLog, "alloc new m_voxel_id_array: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);

    // merging with existing sorted voxel array
    time = getCPUTime();
    thrust::merge_by_key(m_voxel_id_array.begin(), m_voxel_id_array.end(), d_new_voxel_ids.begin(),
                         d_new_voxel_ids.end(), m_voxel_type_flags.begin(), d_voxelTypeFlags.begin(),
                         new_voxel_id_array.begin(), new_voxelTypeFlags.begin());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    LOGGING_INFO(OctreeLog, "thrust::merge: " <<  timeDiff(time, getCPUTime()) << " ms" << endl);

    m_voxel_id_array.swap(new_voxel_id_array);
    m_voxel_type_flags.swap(new_voxelTypeFlags);
  }

  VoxelID* getDevicePtr()
  {
    return D_PTR(m_voxel_id_array);
  }

  VoxelTypeFlags<VTF_SIZE>* getFlagsDevicePtr()
  {
    return D_PTR(m_voxel_type_flags);
  }

  std::size_t size()
  {
    assert(m_voxel_id_array.size() == m_voxel_type_flags.size());
    return m_voxel_id_array.size();
  }

};

}  // end of ns
}  // end of ns

#endif /* VOXELLIST_H_ */
