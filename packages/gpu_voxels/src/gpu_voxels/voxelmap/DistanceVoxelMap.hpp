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
 * \author  Christian Juelg
 * \date    2015-08-19
 *
 * This file wraps code relating to the Parallel Banding Algorithm (PBA)
 * implementation by Cao Thanh Tung from National University of Singapore.
 * That code can be found in the files kernels/VoxelMapOperationsPBA.h
 * and kernels/VoxelMapOperationsPBA.hpp.
 */
//----------------------------------------------------------------------
#ifndef DISTANCEVOXELMAP_HPP
#define DISTANCEVOXELMAP_HPP

#include <gpu_voxels/voxelmap/DistanceVoxelMap.h>
#include <gpu_voxels/voxelmap/TemplateVoxelMap.hpp>

#include <gpu_voxels/voxel/DistanceVoxel.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>

#include <gpu_voxels/logging/logging_gpu_voxels.h>

#ifdef IC_PERFORMANCE_MONITOR
  #include "icl_core_performance_monitor/PerformanceMonitor.h"
#endif

#include <boost/shared_ptr.hpp>

namespace gpu_voxels {
namespace voxelmap {

// uninitialized_allocator is a NO-OP allocator to be used with thrust::device_vector
// see http://stackoverflow.com/questions/16389662/how-to-avoid-default-construction-of-elements-in-thrustdevice-vector
// see https://github.com/thrust/thrust/blob/master/examples/uninitialized_vector.cu
//TODO: move to a helper file, e.g. helpers/thrust_helpers.h/hpp
template<typename T>
struct uninitialized_allocator : thrust::device_malloc_allocator<T>
{
  // note that construct is annotated as
  // a __host__ __device__ function
  __host__ __device__
  void construct(const T *p) const
  {
    // no-op
  }
};

DistanceVoxelMap::DistanceVoxelMap(const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dim, voxel_side_length, map_type)
{

}

DistanceVoxelMap::DistanceVoxelMap(Voxel* dev_data, const Vector3ui dim, const float voxel_side_length, const MapType map_type) :
    Base(dev_data, dim, voxel_side_length, map_type)
{

}

size_t DistanceVoxelMap::collideWithTypes(const GpuVoxelsMapSharedPtr other, BitVectorVoxel&  meanings_in_collision, float coll_threshold, const Vector3ui &offset) {
  //NOP
  return 0;
  //TODO: debug output. nonsense operation or should we actually interpret distance > 0 as free and distance <= 0 as occupied?
}

bool DistanceVoxelMap::insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud *robot_links,
                                                                  const std::vector<BitVoxelMeaning>& voxel_meanings,
                                                                  const std::vector<BitVector<BIT_VECTOR_LENGTH> >& collision_masks,
                                                                  BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
{
  LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
  return true;
}

void DistanceVoxelMap::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
{
  //NOP

  // TODO: maybe clearMap instead? ProbVoxelMap does this
    
  // TODO: printf or make functional
}

bool DistanceVoxelMap::mergeOccupied(const boost::shared_ptr<ProbVoxelMap> other, const Vector3ui &voxel_offset, float occupancy_threshold) {
  boost::lock(this->m_mutex, other->m_mutex);
  lock_guard guard(this->m_mutex, boost::adopt_lock);
  lock_guard guard2(other->m_mutex, boost::adopt_lock);

  //TODO: ensure this->getDimensions == other->getDimensions

//  TODO: add optimized version if offset is zero? if(voxel_offset != Vector3ui())

  //transform (countingiterator(0, this->getVoxelMapSize()), constant iterator(dimensions), this->getDeviceDataPtr() (+0, +getvoxelmapsize)
  thrust::transform_if(
        thrust::device_system_tag(),

        thrust::make_zip_iterator( thrust::make_tuple(other->getDeviceDataPtr(),
                                                      thrust::counting_iterator<uint>(0) )),

        thrust::make_zip_iterator( thrust::make_tuple(other->getDeviceDataPtr() + this->getVoxelMapSize(),
                                                      thrust::counting_iterator<uint>(this->getVoxelMapSize()) )),

        this->getDeviceDataPtr(),

        mergeOccupiedOperator(this->m_dim, voxel_offset),

        probVoxelOccupied(ProbabilisticVoxel::floatToProbability(occupancy_threshold))
      );

  return true;
}

/**
 * cjuelg: jump flood distances, obstacle vectors
 */
void DistanceVoxelMap::jumpFlood3D(int block_size, int debug, bool logging_reinit) {

  if (this->m_dim.x % 2 || this->m_dim.y % 2)
  {
    LOGGING_ERROR(VoxelmapLog, "jumpFlood3D: dimX and dimY cannot be odd numbers" << endl);
    return;
  }

#ifdef IC_PERFORMANCE_MONITOR
  if (logging_reinit) PERF_MON_INITIALIZE(10, 100);
  if (debug) PERF_MON_START("kerneltimer");
#endif

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  DistanceVoxel* temp_buffer;
  HANDLE_CUDA_ERROR(cudaMalloc(&temp_buffer, this->getMemoryUsage()));
  //  HANDLE_CUDA_ERROR(cudaMemset(temp_buffer, 0, this->getMemoryUsage())); //use fill(uninitialised) instead

  DistanceVoxel* buffers[2];
  buffers[0] = this->m_dev_data;
  buffers[1] = temp_buffer;

  //  thrust::device_ptr<DistanceVoxel> original_begin_3d(this->m_dev_data);
  //  thrust::device_ptr<DistanceVoxel> original_end_3d(original_begin_3d + this->getVoxelMapSize());
  //  DistanceVoxelMap::pba_transform(original_begin_3d, original_end_3d, original_begin_3d); //obstacles have (own_coords), 0 format; all else are setPBAUninitialised

  thrust::device_ptr<DistanceVoxel> temp_begin_3d(temp_buffer);
  thrust::device_ptr<DistanceVoxel> temp_end_3d(temp_begin_3d + this->getVoxelMapSize());
  DistanceVoxel pba_uninitialised_voxel;
  pba_uninitialised_voxel.setPBAUninitialised();
  thrust::fill(temp_begin_3d, temp_end_3d, pba_uninitialised_voxel);

  int output_buffer_idx = 1;

  int grid_size = (this->getVoxelMapSize() + block_size - 1) / block_size; //round up
  if (debug) LOGGING_INFO(VoxelmapLog, "grid: " << grid_size << ", block: " << block_size << endl);

  int32_t starting_step = (max(this->m_dim.x, max(this->m_dim.y, this->m_dim.z)) + 1) / 2;
  for (int32_t step_width = starting_step; step_width > 0; step_width /= 2) {

    kernelJumpFlood3D
        <<< grid_size, block_size >>>
        (
          buffers[1 - output_buffer_idx], buffers[output_buffer_idx], this->m_dim, step_width
        );
    CHECK_CUDA_ERROR();

    output_buffer_idx = 1 - output_buffer_idx;
  }
  cudaDeviceSynchronize();

  if (output_buffer_idx != 1) { //odd number of loop iterations -> need to copy buffer to dev_data
    if (debug) LOGGING_INFO(VoxelmapLog, "jumpFlood: memcpy temp_buffer" << endl);
    //optimise: don't copy, instead exchange this->m_dev_data and temp_buffer, free original buffer
    cudaMemcpy(this->m_dev_data, temp_buffer, this->getMemoryUsage(), cudaMemcpyDeviceToDevice);
  }

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaFree(temp_buffer));

#ifdef IC_PERFORMANCE_MONITOR
  if (debug) PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "jumpFlood3D done");
  //  PERF_MON_SUMMARY_ALL_INFO;
#endif
}

void DistanceVoxelMap::exactDistances3D(std::vector<Vector3f>& points) {
#ifdef IC_PERFORMANCE_MONITOR
  PERF_MON_INITIALIZE(10, 100);
  PERF_MON_START("kerneltimer");
#endif

//  if (points.size() > 200*1000) { //should be enough for
//    LOGGING_ERROR(VoxelmapLog, "exactDistances3D: too many obstacles, aborting brute-force comparison" << endl);
//    return;
//  }

// copy points to the gpu
  Vector3f* d_points;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_points, points.size() * sizeof(Vector3f)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_points, &points[0], points.size() * sizeof(Vector3f), cudaMemcpyHostToDevice));

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  dim3 blocks(this->m_blocks);
  uint total_threads = cMAX_THREADS_PER_BLOCK * cMAX_NR_OF_BLOCKS;
  if (total_threads > getVoxelMapSize()) {
    blocks.x = std::max(1u, ((total_threads / cMAX_THREADS_PER_BLOCK)  + m_dim.z - 1) / m_dim.z);
    blocks.y = std::max(1u, ((total_threads / cMAX_THREADS_PER_BLOCK)  + blocks.x - 1) / blocks.x);
  }

  LOGGING_INFO(VoxelmapLog, "#obstacles: " << points.size() << endl);
  LOGGING_INFO(VoxelmapLog, "grid: " << blocks.x << "x" << blocks.y << ", cMAX_NR_OF_BLOCKS: " << cMAX_NR_OF_BLOCKS << " threads: " << this->m_threads << endl);

  size_t dynamic_shared_mem_size = sizeof(Vector3ui) * cMAX_THREADS_PER_BLOCK;
  kernelExactDistances3D
    <<< blocks, this->m_threads, dynamic_shared_mem_size >>>
    (
      this->m_dev_data, this->m_dim,
      this->m_voxel_side_length, d_points, points.size()
    );
  CHECK_CUDA_ERROR();

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  HANDLE_CUDA_ERROR(cudaFree(d_points));

#ifdef IC_PERFORMANCE_MONITOR
    PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "exactDistances3D done");
//  PERF_MON_SUMMARY_ALL_INFO;
#endif
}

/**
 * @brief DistanceVoxelMap::parallelBanding3D
 * @param height
 *
 *
 * 3D process:
 *
 ************* Compute along Z axis *************
 // --> (X, Y, Z)
 pba3DColorZAxis(m1);

 ************* Compute along Y axis *************
 // --> (X, Y, Z)
 pba3DComputeProximatePointsYAxis(m2);
 pba3DColorYAxis(m3);

 // --> (Y, X, Z)
 pba3DTransposeXY();

 ************** Compute along X axis *************
 // Compute X axis
 pba3DComputeProximatePointsYAxis(m2);
 pba3DColorYAxis(m3);

 // --> (X, Y, Z)
 pba3DTransposeXY();

 *
 */
void DistanceVoxelMap::parallelBanding3D(uint32_t m1, uint32_t m2, uint32_t m3, uint32_t arg_m1_blocksize, uint32_t arg_m2_blocksize, uint32_t arg_m3_blocksize, bool detailtimer) {

  if (this->m_dim.x != this->m_dim.y || this->m_dim.x % 64)
  {
    LOGGING_ERROR(VoxelmapLog, "parallelBanding3D: dimX and dimY must be equal; they also must be divisible by 64" << endl);
    //return; //TODO: check whether this is the right check; why not 32?
  }

  bool sync_always = detailtimer;
  if (sync_always); //ifndef IC_PERFORMANCE_MONITOR there would be a compiler warning otherwise

  //optimise m1,m2,m3; m3 is especially detrimental? (increases divergence)
  // m2, m3 works on dim.y first, then dim.x after transpose
  m1 = max(1, min(m1, this->m_dim.z)); //band count in phase1
  m2 = max(1, min(m2, this->m_dim.y)); //band count in phase2
  m3 = max(1, min(m3, this->m_dim.y)); //band count in phase3

  // ensure m_dim.z is multiple of m1
  if (this->m_dim.z % m1) {
    LOGGING_WARNING(VoxelmapLog, "PBA: m1 does not cleanly divide m_dim.z: " << this->m_dim.z << "%" << m1 << " = " << (this->m_dim.z % m1) << ", reverting to default m1 = 1" << endl);
    m1 = 1;
  }

  // ensure m_dim.x and m_dim.y are multiples of m2
  if ((this->m_dim.x % m2) || (this->m_dim.y % m2)) {
    LOGGING_WARNING(VoxelmapLog, "PBA: m2 does not cleanly divide m_dim.x and m_dim.y: " << this->m_dim.x << "%" << m2 << " = " << (this->m_dim.x % m2) << ", " << this->m_dim.y << "%" << m2 << " = " << (this->m_dim.y % m2) <<  ", reverting to default m2 = 1" << endl);
    m2 = 1;
  }

  // ensure m_dim.x and m_dim.y are multiples of m3
  if ((this->m_dim.x % m3) || (this->m_dim.y % m3)) {
    LOGGING_WARNING(VoxelmapLog, "PBA: m3 does not cleanly divide m_dim.x and m_dim.y: " << this->m_dim.x << "%" << m3 << " = " << (this->m_dim.x % m3) << ", " << this->m_dim.y << "%" << m3 << " = " << (this->m_dim.y % m3) <<  ", reverting to default m3 = 1" << endl);
    m3 = 1;
  }

  LOGGING_DEBUG(VoxelmapLog, "PBA: m1: " << m1 << ", m2: " << m2 << ", m3: " << m3 << ", detailtimer: " << detailtimer << endl);

#ifdef IC_PERFORMANCE_MONITOR
  if (detailtimer) PERF_MON_START("detailtimer");
#endif

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

#ifdef IC_PERFORMANCE_MONITOR
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D warmup sync");
  PERF_MON_START("pbatimer");
#endif

//  uint32_t layer_size = this->m_dim.x * this->m_dim.y;

//int debugging = 0;
//LOGGING_INFO(VoxelmapLog, "thrust call " << (debugging++) << endl);
  thrust::device_ptr<DistanceVoxel> original_begin_3d(this->m_dev_data);
  thrust::device_ptr<DistanceVoxel> original_end_3d(this->m_dev_data + this->m_voxelmap_size);

  //optimise: use uint3 or ushort3 for initial and distance_map; check for usage of distance in phase1-3. add distance to voxelmap either after phase3 or: in phase3, write directly to DVM, including distance

  //optimise by re-using original_begin and _end for initial_map
  thrust::device_vector<DistanceVoxel> initial_map(original_begin_3d, original_end_3d);
//    thrust::device_vector<DistanceVoxel, uninitialized_allocator<DistanceVoxel> > initial_map(this->getVoxelMapSize());
  //  thrust::copy(original_begin_3d, original_end_3d, initial_map.begin());
  //  thrust::replace_if(voxel_begin, voxel_end, typename DistanceVoxel::is_obstacle(), pba_uninitialised_voxel);

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D initialisation device_vector created");
#endif

//  //TODO: uninit voxels are already PBA_UNINIT; will only change distance from -1 to 0 in obstacles!
//  DistanceVoxelMap::pba_transform
//      (original_begin_3d, original_end_3d, initial_map.begin()); //in_begin, in_end, output_begin, unary_op

//#ifdef IC_PERFORMANCE_MONITOR
//  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D initialisation after transform");
//#endif

  //prefill this map with pba_uninitialised_voxel
//  DistanceVoxel pba_uninitialised_voxel;
//  pba_uninitialised_voxel.setPBAUninitialised();
  //TODO: compare speed to thrust::device_vector<DistanceVoxel> distance_map(this->m_voxelmap_size, pba_uninitialised_voxel);
//  thrust::device_vector<DistanceVoxel> distance_map(this->m_voxelmap_size);
  thrust::device_ptr<DistanceVoxel> distance_map_begin = original_begin_3d;
//  thrust::device_vector<DistanceVoxel> distance_map(original_begin_3d, original_end_3d);

//  thrust::device_ptr<DistanceVoxel> distance_map_end = original_end_3d;
//  thrust::fill(distance_map_begin, distance_map_end, pba_uninitialised_voxel);

#ifdef IC_PERFORMANCE_MONITOR
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D initialisation done");
  PERF_MON_PRINT_AND_RESET_INFO_P("pbatimer", "parallelBanding3D init done", "pbaprefix");
#endif

  // PBA phase 1
  //     optimise: could work as series of simple transforms in one array

  //TODO: ensure blocksize divides m_dim.* evenly

  //in total m1*dim.x*dim.y threads
  //within warp threads should access x-neighbors
  dim3 m1_block_size(min(arg_m1_blocksize, this->m_dim.x)); // optimize blocksize
  dim3 m1_grid_size(this->m_dim.x / m1_block_size.x, this->m_dim.y, m1); //m1 bands

  if (this->getDimensions().x < m1_block_size.x) {
    //TODO: check for phase1-3 that m1-3 and blocksizes are always safe and kernels will not cause memory acces violations
    LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, "ERROR: PBA require dimensions.x >= PBA_BLOCKSIZE (" << PBA_BLOCKSIZE << ")" << endl);
  }

  //flood foward and backward within bands
  //there are m1 vertical bands
  //TODO optimise: could work in-place
  kernelPBAphase1FloodZ
      <<< m1_grid_size, m1_block_size >>>
      (distance_map_begin, distance_map_begin, this->m_dim, this->m_dim.z / m1); //distance_map is output
  CHECK_CUDA_ERROR();
  // -> blöcke enthalten gelbe vertikale balken, solange min 1 obstacle enthalten

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 1 floodZ done");
#endif

  //pass information between bands
  //optimise: propagate and update could be in same kernel
  if (m1 > 1) {
    kernelPBAphase1PropagateInterband
        <<< m1_grid_size, m1_block_size >>>
        (distance_map_begin, initial_map.begin(), this->m_dim, this->m_dim.z / m1); //buffer b to a
    CHECK_CUDA_ERROR();
    // -> initial_map enthält obstacle infos und interband head/tail infos
  }

#ifdef IC_PERFORMANCE_MONITOR
  if (m1 > 1) {
    if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 1 interband done");
  }
#endif

  if (m1 > 1) {
    kernelPBAphase1Update
          <<< m1_grid_size, m1_block_size >>>
          (initial_map.begin(), distance_map_begin, this->m_dim, this->m_dim.z / m1); //buffer to b; a is Links (top,bottom), b is Color (voxel)
    CHECK_CUDA_ERROR();
  }
  // end of phase 1: distance_map contains the S_ij obstacle information

#ifdef IC_PERFORMANCE_MONITOR
  if (m1 > 1) {
    if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 1 update done");
  }
#endif

  // compute proximate points locally in each band
  dim3 m2_block_size(min(arg_m2_blocksize, min(this->m_dim.x, this->m_dim.y))); // optimize blocksize
  dim3 m2_grid_size = dim3(this->m_dim.x / m2_block_size.x, m2, m_dim.z); // m2 bands per column

  if ((this->getDimensions().x < arg_m2_blocksize) || (this->getDimensions().y < arg_m2_blocksize)) {
    //TODO: check for phase1-3 that m1-3 and blocksizes are always safe and kernels will not cause memory acces violations
    LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, "ERROR: PBA requires dimensions.x and .y >= arg_m2_blocksize (" << arg_m2_blocksize << ")" << endl);
  }

  kernelPBAphase2ProximateBackpointers
      <<< m2_grid_size, m2_block_size >>>
     (distance_map_begin, initial_map.begin(), this->m_dim, this->m_dim.y / m2); //output stack/singly linked list with backpointers; some elements are skipped
  CHECK_CUDA_ERROR();

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 2 backpointer done");
#endif

  // distance_map will be shadowed by an array of int16 for CreateForward and MergeBands
//  thrust::device_ptr<pba_fw_ptr_t> forward_ptrs_begin((pba_fw_ptr_t*)(distance_map_begin.get()));
  thrust::device_ptr<pba_fw_ptr_t> forward_ptrs_begin((pba_fw_ptr_t*)(distance_map_begin.get()));

  if (m2 > 1) {
    kernelPBAphase2CreateForwardPointers
        <<< m2_grid_size, m2_block_size >>>
        (initial_map.begin(), forward_ptrs_begin, this->m_dim, this->m_dim.y / m2); //read stack, write forward pointers
    CHECK_CUDA_ERROR();
  }

#ifdef IC_PERFORMANCE_MONITOR
  if (m2 > 1) {
    if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 2 forward done");
  }
#endif

  // repeatedly merge two bands into one
  for (int band_count = m2; band_count > 1; band_count /= 2) {
    dim3 m2_merge_grid_size = dim3(this->m_dim.x / m2_block_size.x, band_count / 2, this->m_dim.z);

    kernelPBAphase2MergeBands
        <<< m2_merge_grid_size, m2_block_size >>>
        (initial_map.begin(), forward_ptrs_begin, this->m_dim, this->m_dim.y / band_count); //update both stack and forward_ptrs
    CHECK_CUDA_ERROR();

    if (detailtimer) LOGGING_INFO(VoxelmapLog, "kernelPBAphase2MergeBands finished merging with band_size " << (this->m_dim.y / band_count) << endl);

#ifdef IC_PERFORMANCE_MONITOR
    if (m2 > 1) {
      if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
      if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 2 merge iteration done");
    }
#endif
  }
  // end of phase 2: initial_ contains P_i information; y coordinates were replaced by back-pointers; y coordinate is implicitly equal to voxel position.y

  // TODO: benchmark and/or delete texture usage: (initialResDesc, texDesc and initialTexObj)
  //TODO: use template specialisation to implement; run once with and without textures

  // Specify texture
  struct cudaResourceDesc initialResDesc;
  memset(&initialResDesc, 0, sizeof(initialResDesc));
  initialResDesc.resType = cudaResourceTypeLinear;
  initialResDesc.res.linear.devPtr = thrust::raw_pointer_cast(initial_map.data());
  initialResDesc.res.linear.sizeInBytes = initial_map.size()*sizeof(int);

  //TODO!
  initialResDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
  initialResDesc.res.linear.desc.x = 32; // bits per channel

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture object
  cudaTextureObject_t initialTexObj = 0;
  cudaCreateTextureObject(&initialTexObj, &initialResDesc, &texDesc, NULL);



//  // phase 3: read from input_, write to distance_map
//  //optimise: scale PBA_M3_BLOCKX to m3; PBA_M3_BLOCKX*m3 should not be too small
  dim3 m3_block_size(min(arg_m3_blocksize, min(this->m_dim.x, this->m_dim.y)), m3); // y bands; block_size threads in total
  dim3 m3_grid_size = dim3(this->m_dim.x / m3_block_size.x, 1, this->m_dim.z);

  if ((this->getDimensions().x < arg_m3_blocksize) || (this->getDimensions().y < arg_m3_blocksize)) {
    //TODO: check for phase1-3 that m1-3 and blocksizes are always safe and kernels will not cause memory acces violations
    LOGGING_ERROR_C(VoxelmapLog, DistanceVoxelMap, "ERROR: PBA requires dimensions.x and .y >= arg_m2_blocksize (" << arg_m2_blocksize << ")" << endl);
  }
  //distance map is write-only during phase3
  kernelPBAphase3Distances
      <<< m3_grid_size, m3_block_size >>>
        (initialTexObj, distance_map_begin, this->m_dim);
  CHECK_CUDA_ERROR();
      //  (initial_map.begin(), distance_map_begin, this->m_dim);
  // phase 3 done: distance_map contains final result

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first phase 3 done");
#endif

  // transpose x/y within every z-layer; need to transpose obstacle coordinates as well
  // transpose in-place to reuse input/ouput scheme of phase 2&3
  // optimise: check bare transpose performance in-place vs non-inplace
  //TODO: ensure m_dim x/y divisible by PBA_TILE_DIM
  dim3 transpose_block(PBA_TILE_DIM, PBA_TILE_DIM);
  dim3 transpose_grid(this->m_dim.x / transpose_block.x, this->m_dim.y / transpose_block.y, this->m_dim.z); //maximum blockDim.y/z is 64K
  kernelPBA3DTransposeXY<<<transpose_grid, transpose_block>>>
                        (distance_map_begin); //optimise: remove thrust wrapper?
  CHECK_CUDA_ERROR();

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D first transpose done");
#endif

  //second phase2&3:
  // setze m2_grid_size erneut!
  // compute proximate points locally in each band

  kernelPBAphase2ProximateBackpointers
      <<< m2_grid_size, m2_block_size >>>
     (distance_map_begin, initial_map.begin(), this->m_dim, this->m_dim.y / m2); //output stack/singly linked list with backpointers; some elements are skipped
  CHECK_CUDA_ERROR();

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase 2 backpointers done");
#endif


  if (m2 > 1) {
    kernelPBAphase2CreateForwardPointers
        <<< m2_grid_size, m2_block_size >>>
        (initial_map.begin(), forward_ptrs_begin, this->m_dim, this->m_dim.y / m2); //read stack, write forward pointers
    CHECK_CUDA_ERROR();
  }

#ifdef IC_PERFORMANCE_MONITOR
  if (m2 > 1) {
    if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase 2 forwardpointers done");
  }
#endif


  // repeatedly merge two bands into one
  for (int band_count = m2; band_count > 1; band_count /= 2) {
    dim3 m2_merge_grid_size = dim3(this->m_dim.x / m2_block_size.x, band_count / 2, this->m_dim.z);
    kernelPBAphase2MergeBands
        <<< m2_merge_grid_size, m2_block_size >>>
        (initial_map.begin(), forward_ptrs_begin, this->m_dim, this->m_dim.y / band_count); //update both stack and forward_ptrs
    CHECK_CUDA_ERROR();

    if (detailtimer) LOGGING_INFO(VoxelmapLog, "kernelPBAphase2MergeBands finished merging with band_size " << (this->m_dim.y / band_count) << endl);


#ifdef IC_PERFORMANCE_MONITOR
    if (m2 > 1) {
      if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
      if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase2 merge iteration done");
    }
#endif
  }
  // end of phase 2: initial_ contains P_i information; y coordinates were replaced by back-pointers; y coordinate is implicitly equal to voxel position.y
  // phase 3: read from input_, write to distance_map
  //optimise: scale PBA_M3_BLOCKX to m3; PBA_M3_BLOCKX*m3 should not be too small
  kernelPBAphase3Distances
      <<< m3_grid_size, m3_block_size >>>
//      (initial_map.begin(), distance_map_begin, this->m_dim);
      (initialTexObj, distance_map_begin, this->m_dim);
  CHECK_CUDA_ERROR();
  // phase 3 done: distance_map contains final result

  //second phase2&3 done

#ifdef IC_PERFORMANCE_MONITOR
  if (sync_always) HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second phase3 done");
#endif

  kernelPBA3DTransposeXY<<<transpose_grid, transpose_block>>>
                        (distance_map_begin);
  CHECK_CUDA_ERROR();

//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
//  //copy back distance_map to m_dev_data
//  thrust::copy(distance_map_begin, distance_map_end, original_begin_3d);
//  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

  // Destroy texture object
  cudaDestroyTextureObject(initialTexObj);

#ifdef IC_PERFORMANCE_MONITOR
  if (detailtimer) PERF_MON_PRINT_AND_RESET_INFO("detailtimer", "parallelBanding3D second transpose done");
  PERF_MON_PRINT_AND_RESET_INFO_P("pbatimer", "parallelBanding3D compute done", "pbaprefix");
//  PERF_MON_SUMMARY_ALL_INFO;
#endif

}

/**
 * @brief DistanceVoxelMap::clone clone other DVM
 * @param other DVM to be cloned
 */
void DistanceVoxelMap::clone(DistanceVoxelMap& other) {
  thrust::device_ptr<DistanceVoxel> other_begin(other.m_dev_data);
  thrust::device_ptr<DistanceVoxel> other_end(other_begin + other.getVoxelMapSize());
  thrust::device_ptr<DistanceVoxel> this_begin(this->m_dev_data);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  thrust::copy(other_begin, other_end, this_begin);
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

void DistanceVoxelMap::fill_pba_uninit() {
  this->fill_pba_uninit(*this); //obstacles have (own_coords,0) format; all else are setPBAUninitialised
}

void DistanceVoxelMap::fill_pba_uninit(DistanceVoxelMap& out) {
  thrust::device_ptr<DistanceVoxel> original_begin(out.m_dev_data);
  thrust::device_ptr<DistanceVoxel> original_end(original_begin + out.getVoxelMapSize());
  DistanceVoxel dv_uninit;
  dv_uninit.setPBAUninitialised();
  thrust::fill(original_begin, original_end, dv_uninit);
}

void DistanceVoxelMap::init_floodfill(free_space_t* dev_distances, manhattan_dist_t* dev_manhattan_distances, uint robot_radius) {
  // thrust transform pbaDistanceVoxmap->getDeviceDataPtr() to byte[] (round down to 0..255, cap at 255; could even parameterize on robot size and create boolean

  LOGGING_INFO(VoxelmapLog, "init_floodfill for " << this->getVoxelMapSize() << " voxels" << endl);
//  LOGGING_INFO(VoxelmapLog, "in_begin: " << dev_distances << endl);
//  LOGGING_INFO(VoxelmapLog, "in_end: " << (dev_distances + this->getVoxelMapSize()) << endl);
//  LOGGING_INFO(VoxelmapLog, "out_begin: " << dev_manhattan_distances << endl);

  thrust::transform(thrust::device_system_tag(),
                    dev_distances,
                    dev_distances + this->getVoxelMapSize(),
                    dev_manhattan_distances,
                    DistanceVoxel::init_floodfill_distance(robot_radius));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

void DistanceVoxelMap::extract_distances(free_space_t* dev_distances, int robot_radius) const
{
  thrust::device_ptr<DistanceVoxel> dev_voxel_begin(this->m_dev_data);
  thrust::device_ptr<DistanceVoxel> dev_voxel_end(dev_voxel_begin + this->getVoxelMapSize());
  thrust::device_ptr<free_space_t> dev_free_space_begin(dev_distances);

  // thrust transform pbaDistanceVoxmap->getDeviceDataPtr() to byte[] (round down to 0..255, cap at 255; could even parameterize on robot size and create boolean
  thrust::counting_iterator<int> count_start(0);
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(dev_voxel_begin, count_start)),
                    thrust::make_zip_iterator(thrust::make_tuple(dev_voxel_end, count_start + this->getVoxelMapSize())),
                    dev_free_space_begin,
                    DistanceVoxel::extract_byte_distance(Vector3i(m_dim), robot_radius));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

DistanceVoxel::pba_dist_t DistanceVoxelMap::getSquaredObstacleDistance(const Vector3ui& pos) {
  return this->getSquaredObstacleDistance(pos.x, pos.y, pos.z);
}

DistanceVoxel::pba_dist_t DistanceVoxelMap::getSquaredObstacleDistance(uint x, uint y, uint z) {
  // get voxel from device memory
  DistanceVoxel dv;
  cudaMemcpy(&dv, getVoxelPtr(m_dev_data, m_dim, x, y, z), sizeof(DistanceVoxel), cudaMemcpyDeviceToHost);
  return dv.squaredObstacleDistance(Vector3i(x, y, z));
}

DistanceVoxel::pba_dist_t DistanceVoxelMap::getObstacleDistance(const Vector3ui& pos) {
  return sqrt(this->getSquaredObstacleDistance(pos.x, pos.y, pos.z));
}

DistanceVoxel::pba_dist_t DistanceVoxelMap::getObstacleDistance(uint x, uint y, uint z) {
  return sqrt(this->getSquaredObstacleDistance(x, y, z));
}

struct SquaredDistanceFunctor
{
  Vector3ui dims;

  __host__ __device__
  SquaredDistanceFunctor(Vector3ui dims) : dims(dims) {}
  
  __host__ __device__
  DistanceVoxel::pba_dist_t operator()(thrust::tuple<DistanceVoxel, uint> t)
  {
    DistanceVoxel voxel = thrust::get<0>(t);
    uint linear_id = thrust::get<1>(t);
    Vector3ui position = mapToVoxels(linear_id, dims);
    return voxel.squaredObstacleDistance(Vector3i(position.x, position.y, position.z));
  }
};

void DistanceVoxelMap::getSquaredDistancesToHost(std::vector<uint>& indices, std::vector<DistanceVoxel::pba_dist_t>& output)
{
  // copy indices to device
  thrust::device_vector<uint> dev_indices(indices);
  
  // allocate device output memory
  thrust::device_vector<DistanceVoxel::pba_dist_t> dev_output(indices.size());
  
  //call dev function
  getSquaredDistances(&(*dev_indices.begin()), &(*dev_indices.end()), &(*dev_output.begin()));
  
  //copy output to std_vector
  thrust::copy(dev_output.begin(), dev_output.end(), output.begin());
}

void DistanceVoxelMap::getSquaredDistances(thrust::device_ptr<uint> dev_indices_begin, thrust::device_ptr<uint> dev_indices_end, thrust::device_ptr<DistanceVoxel::pba_dist_t> dev_output)
{  
  //get the Voxels corresponding to the selected indices
  thrust::device_vector<DistanceVoxel> dev_voxels(dev_indices_end - dev_indices_begin);
  gatherVoxelsByIndex(dev_indices_begin, dev_indices_end, dev_voxels.data());

  // extract the distances for the indexed voxels
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(dev_voxels.begin(), dev_indices_begin)),
                    thrust::make_zip_iterator(thrust::make_tuple(dev_voxels.end(), dev_indices_end)),
                    dev_output,
                    SquaredDistanceFunctor(m_dim));
}

struct DistanceFunctor
{
  Vector3ui dims;

  __host__ __device__
  DistanceFunctor(Vector3ui dims) : dims(dims) {}
  
  __host__ __device__
  DistanceVoxel::pba_dist_t operator()(thrust::tuple<DistanceVoxel, uint> t)
  {
    DistanceVoxel voxel = thrust::get<0>(t);
    uint linear_id = thrust::get<1>(t);
    Vector3ui position = mapToVoxels(linear_id, dims);
    return sqrtf(voxel.squaredObstacleDistance(Vector3i(position.x, position.y, position.z)));
  }
};

void DistanceVoxelMap::getDistancesToHost(std::vector<uint>& indices, std::vector<DistanceVoxel::pba_dist_t>& output)
{
  // copy indices to device
  thrust::device_vector<uint> dev_indices(indices);
  
  // allocate device output memory
  thrust::device_vector<DistanceVoxel::pba_dist_t> dev_output(indices.size());
  
  //call dev function
  getDistances(&(*dev_indices.begin()), &(*dev_indices.end()), &(*dev_output.begin()));
  
  //copy output to std_vector
  thrust::copy(dev_output.begin(), dev_output.end(), output.begin());
}

void DistanceVoxelMap::getDistances(thrust::device_ptr<uint> dev_indices_begin, thrust::device_ptr<uint> dev_indices_end, thrust::device_ptr<DistanceVoxel::pba_dist_t> dev_output)
{  
  //get the Voxels corresponding to the selected indices
  thrust::device_vector<DistanceVoxel> dev_voxels(dev_indices_end - dev_indices_begin);
  gatherVoxelsByIndex(dev_indices_begin, dev_indices_end, dev_voxels.data());

  // extract the distances for the indexed voxels
  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(dev_voxels.begin(), dev_indices_begin)),
                    thrust::make_zip_iterator(thrust::make_tuple(dev_voxels.end(), dev_indices_end)),
                    dev_output,
                    DistanceFunctor(m_dim));
}

/**
 * @brief DistanceVoxelMap::differences3D
 * @param other
 * @return
 */
DistanceVoxel::accumulated_diff DistanceVoxelMap::differences3D(const boost::shared_ptr<DistanceVoxelMap> other, int debug, bool logging_reinit) {

#ifdef IC_PERFORMANCE_MONITOR
  if (logging_reinit)
  {
    PERF_MON_INITIALIZE(10, 100);
  }
  PERF_MON_START("kerneltimer");
#endif

  thrust::device_ptr<DistanceVoxel> voxel_begin(this->m_dev_data);
  thrust::device_ptr<DistanceVoxel> voxel_end(this->m_dev_data + this->m_voxelmap_size);
  thrust::device_ptr<DistanceVoxel> other_voxel_begin(other->m_dev_data);
  thrust::device_ptr<DistanceVoxel> other_voxel_end(other->m_dev_data + this->m_voxelmap_size);

  // use thrust to reduce both mindist and count?
  DistanceVoxel::accumulated_diff init;

  //TODO: remove debug code

  //debug
  thrust::device_vector<DistanceVoxel> dev_map(voxel_begin, voxel_end);
  thrust::host_vector<DistanceVoxel> host_map = dev_map; //optimise: copy only the data that will be compared

  thrust::device_vector<DistanceVoxel> dev_other_map(other_voxel_begin, other_voxel_end);
  thrust::host_vector<DistanceVoxel> host_other_map = dev_other_map; //optimise: copy only the data that will be compared

//  LOGGING_INFO(VoxelmapLog, "map size: " << host_map.size() << ", host_other_map size: " << host_other_map.size() << endl);

//#ifdef IC_PERFORMANCE_MONITOR
//    PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "differences3D 2xmemcpy done");
//#endif

  //debug
  if (debug > 1) {
//    //TODO: show every 23+24*k-th voxel, in x, y, z dir, including direct neighbors

//    //x dir
//    for (int base = 23; base < (int)this->getDimensions().x; base += 24) {
//      for (int offset = -1; offset <= 1; offset++) {
//        Vector3ui voxelpos = Vector3ui(std::min(base + offset, (int)this->getDimensions().x), 20, 20);
//        ptrdiff_t idx = getVoxelIndex(m_dim, voxelpos);
//        const gpu_voxels::DistanceVoxel& d = dev_map[idx];
//        LOGGING_INFO(VoxelmapLog, "x-dir: map["
//                     << (voxelpos.x) << "/" << (voxelpos.y) << "/" << (voxelpos.z)
//                       << "] " << d << endl);
//      }
//    }

//    //y dir
//    for (int base = 23; base < (int)this->getDimensions().y; base += 24) {
//      for (int offset = -1; offset <= 1; offset++) {
//        Vector3ui voxelpos = Vector3ui(20, std::min(base + offset, (int)this->getDimensions().y), 20);
//        ptrdiff_t idx = getVoxelIndex(m_dim, voxelpos);
//        const gpu_voxels::DistanceVoxel& d = dev_map[idx];
//        LOGGING_INFO(VoxelmapLog, "y-dir: map["
//                     << (voxelpos.x) << "/" << (voxelpos.y) << "/" << (voxelpos.z)
//                       << "] " << d << endl);
//      }
//    }

//    //z dir
//    for (int base = 23; base < (int)this->getDimensions().z; base += 24) {
//      for (int offset = -1; offset <= 1; offset++) {
//        Vector3ui voxelpos = Vector3ui(20, 20, std::min(base + offset, (int)this->getDimensions().z));
//        ptrdiff_t idx = getVoxelIndex(m_dim, voxelpos);
//        const gpu_voxels::DistanceVoxel& d = dev_map[idx];
//        LOGGING_INFO(VoxelmapLog, "z-dir: map["
//                     << (voxelpos.x) << "/" << (voxelpos.y) << "/" << (voxelpos.z)
//                       << "] " << d << endl);
//      }
//    }

    int idx;
  //  idx = 2624010;
    idx = 0;
    LOGGING_INFO(VoxelmapLog, "map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_map[idx] << ", other_map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_other_map[idx] << endl);
    idx = 1;
    LOGGING_INFO(VoxelmapLog, "map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_map[idx] << ", other_map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_other_map[idx] << endl);
    idx = 2;
    LOGGING_INFO(VoxelmapLog, "map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_map[idx] << ", other_map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_other_map[idx] << endl);
  //  idx = 3;
  //  idx = 4;
    idx = getVoxelIndexUnsigned(m_dim, Vector3ui(0, 0, 40));
    LOGGING_INFO(VoxelmapLog, "map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_map[idx] << ", other_map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_other_map[idx] << endl);

    Vector3ui obstacle(10, 10, 40);
    for (int dz = -2; dz <= 2; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        Vector3ui pos = obstacle + Vector3ui(0, dy, dz);
        ptrdiff_t pos_idx = getVoxelIndexUnsigned(m_dim, pos);
        const DistanceVoxel& d = host_map[pos_idx];
        const DistanceVoxel& od = host_other_map[pos_idx];
        LOGGING_INFO(VoxelmapLog, "map["
                     << (pos_idx % this->m_dim.x) << "/" << ((pos_idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (pos_idx/(this->m_dim.x*this->m_dim.y))
                     << "] " << d << ", other_map["
                     << (pos_idx % this->m_dim.x) << "/" << ((pos_idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (pos_idx/(this->m_dim.x*this->m_dim.y))
                     << "] " << od << endl);
      }
    }

    obstacle = Vector3ui(10, 10, 40);
    ptrdiff_t obstacle_idx = getVoxelIndexUnsigned(m_dim, obstacle);
    for (int i = -2; i <= 2; i++) {
      idx = obstacle_idx + i;
      const DistanceVoxel& d = host_map[idx];
      const DistanceVoxel& od = host_other_map[idx];
      LOGGING_INFO(VoxelmapLog, "map["
                   << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                   << "] " << d << ", other_map["
                   << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                   << "] " << od << endl);
    }

    obstacle = Vector3ui(27, 44, 4);
    for (int dz = -1; dz <= 1; dz++) {
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          Vector3ui voxel(max(obstacle.x + dx, 0), max(obstacle.y + dy, 0), max(obstacle.z + dz, 0));
          idx = getVoxelIndexUnsigned(m_dim, voxel);
          const DistanceVoxel& d = host_map[idx];
          const DistanceVoxel& od = host_other_map[idx];
          LOGGING_INFO(VoxelmapLog, "map["
                       << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                       << "] " << d << ", other_map["
                       << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                       << "] " << od << endl);
        }
      }
    }

    idx = getVoxelIndexUnsigned(m_dim, Vector3ui(this->m_dim.x - 1, this->m_dim.y - 1, 40));
    LOGGING_INFO(VoxelmapLog, "map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_map[idx] << ", other_map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_other_map[idx] << endl);

    idx = getVoxelIndexUnsigned(m_dim, Vector3ui(this->m_dim.x - 1, this->m_dim.y - 1, this->m_dim.z - 1));
    LOGGING_INFO(VoxelmapLog, "map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_map[idx] << ", other_map["
                 << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                 << "] " << host_other_map[idx] << endl);
//#ifdef IC_PERFORMANCE_MONITOR
//    PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "differences3D 5 point printout done");
//#endif
  }

  if (debug != 0) {
// no printout
#ifdef IC_PERFORMANCE_MONITOR
    PERF_MON_START("kerneltimer");
#endif

//  //create temporary vector containing diff values
//  thrust::device_vector<double> diffs(this->m_voxelmap_size);
//  thrust::transform
//                  (voxel_begin,
//                   voxel_end,
//                   other_voxel_begin,
//                   diffs.begin(),
//                   typename DistanceVoxel::diff_op());

//  DistanceVoxel::accumulated_diff result =
//      thrust::reduce
//                  (diffs.begin(),
//                   diffs.end(),
//                   init,
//                   typename DistanceVoxel::accumulate_op());

    // count number of voxels that are not uninitialised
    size_t initialised_voxels;
    initialised_voxels = thrust::count_if(voxel_begin, voxel_end, DistanceVoxel::is_initialised());
    LOGGING_INFO(VoxelmapLog, "map has " << initialised_voxels << " initialised voxels out of " << (voxel_end - voxel_begin) <<" voxels." << endl);

    // count number of voxels that are not uninitialised
    initialised_voxels = thrust::count_if(other_voxel_begin, other_voxel_end, DistanceVoxel::is_initialised());
    LOGGING_INFO(VoxelmapLog, "other_map has " << initialised_voxels << " initialised voxels out of " << (other_voxel_end - other_voxel_begin) <<" voxels." << endl);
  }

  thrust::counting_iterator<int> count_start(0);

  DistanceVoxel::accumulated_diff result =
      thrust::inner_product
                  (thrust::make_zip_iterator(thrust::make_tuple(voxel_begin, count_start)),
                   thrust::make_zip_iterator(thrust::make_tuple(voxel_end, count_start + this->m_voxelmap_size)),
                   thrust::make_zip_iterator(thrust::make_tuple(other_voxel_begin, count_start + 0)),
                   init,
                   typename DistanceVoxel::accumulate_op(),
                   DistanceVoxel::diff_op(m_dim));

//  //set obstacle distances to 0
//  thrust::transform(voxel_begin, voxel_end, voxel_begin, typename DistanceVoxel::obstacle_zero_transform()); //problem: subsequent kernels will not find OBSTACLE_DISTANCE any more!

  if (debug != 0) {
#ifdef IC_PERFORMANCE_MONITOR
    PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "differences3D inner product done");
#endif

    //debug
    if ((result.count > 0) && (result.count < 250)) { //don't print differing obstacles at identical distance?
      for (size_t i = 0; i < host_map.size(); i++) {
        const DistanceVoxel& d = host_map[i];
        int32_t dd = d.squaredObstacleDistance(mapToVoxelsSigned(i, m_dim));
        const DistanceVoxel& od = host_other_map[i];
        int32_t odd = od.squaredObstacleDistance(mapToVoxelsSigned(i, m_dim));
        if ((dd != odd) && !((dd + odd == -1) && (dd * odd == 0))) { //don't show anything if the distances are 0 and -1
          int idx = i;
          LOGGING_INFO(VoxelmapLog, "map["
                       << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                       << "] " << d << ", other_map["
                       << (idx % this->m_dim.x) << "/" << ((idx / this->m_dim.x) %(this->m_dim.y)) << "/" << (idx/(this->m_dim.x*this->m_dim.y))
                       << "] " << od << endl);
        }
      }
  //#ifdef IC_PERFORMANCE_MONITOR
  //    PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "differences3D difference printout done");
  //#endif
    }

  }

#ifdef IC_PERFORMANCE_MONITOR
    PERF_MON_PRINT_AND_RESET_INFO("kerneltimer", "differences3D done");
    if (debug != 0) {
      PERF_MON_SUMMARY_ALL_INFO;
    }
#endif

  return result;
}

} // end of namespace voxelmap
} // end of namespace gpu_voxels
#endif // DISTANCEVOXELMAP_HPP

