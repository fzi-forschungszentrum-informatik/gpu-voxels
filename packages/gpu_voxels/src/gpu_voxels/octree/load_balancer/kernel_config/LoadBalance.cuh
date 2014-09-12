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
 * \date    2014-07-24
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_LOAD_BALANCE_CUH_INCLUDED
#define GPU_VOXELS_OCTREE_LOAD_BALANCED_KERNEL_CONFIG_LOAD_BALANCE_CUH_INCLUDED

#include <gpu_voxels/octree/kernels/kernel_common.h>

// thrust
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

namespace gpu_voxels {
namespace NTree {
namespace LoadBalancer {

template<class WorkItem, std::size_t level_count>
__global__ void kernelCountElements(WorkItem* work_stacks, uint32_t* work_stacks_item_count,
                                    const uint32_t stack_size_per_task, uint32_t* item_sums_per_level,
                                    uint32_t* inter_stack_offsets)
{
  __shared__ uint32_t shared_task_sum_array[level_count];

  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t num_threads = blockDim.x;
  const uint32_t num_elements = work_stacks_item_count[task_id];
  const WorkItem* my_work_stack = &work_stacks[task_id * stack_size_per_task];

  if (num_elements > stack_size_per_task)
  {
    printf("Work stack overflow! %u %u\n", num_elements, stack_size_per_task);
    return;
  }

// init shared_task_sum_array
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    shared_task_sum_array[i] = 0;
  __syncthreads();

// count elements per level
  for (uint32_t i = thread_id; i < num_elements; i += num_threads)
  {
    assert(my_work_stack[i].level < level_count);

    // assure sorted by level
    assert((i + 1 == num_elements) || my_work_stack[i].level >= my_work_stack[i + 1].level);

    if (!((i + 1 == num_elements) || my_work_stack[i].level >= my_work_stack[i + 1].level))
    {
      printf("Incorrect stack sorting!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
      //exit(0);
    }
    atomicAdd(&shared_task_sum_array[my_work_stack[i].level], 1);
  }
  __syncthreads();

// store each level consecutive in memory to be able to use the thrust scan without a custom iterator
// start with highest level (reverse level order)
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    item_sums_per_level[gridDim.x * (level_count - 1 - i) + task_id] = shared_task_sum_array[i];
  __syncthreads();

// stack local prefix sum from right (starting at highest level)
  if (thread_id == 0)
  {
    uint32_t sum = 0;
#pragma unroll
    for (int l = level_count - 1; l >= 0; --l)
    {
      const uint32_t tmp = shared_task_sum_array[l];
      shared_task_sum_array[l] = sum;
      sum += tmp;
    }
  }
  __syncthreads();

// store local prefix sum
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    inter_stack_offsets[task_id * level_count + i] = shared_task_sum_array[i];
  __syncthreads();

//  // compute a stack local prefix sum, so each threads can compute it's inter level offset to move the data
//  // Specialize BlockScan for 128 threads on type int
//  typedef cub::BlockScan<uint32_t, TRAVERSAL_THREADS> BlockScan;
//  // Allocate shared memory for BlockScan
//  __shared__ typename BlockScan::TempStorage temp_storage;
//
//  // Collectively compute the block-wide exclusive prefix sum
//  BlockScan(temp_storage).ExclusiveSum(shared_sumArray[threadId],
//                                       interQueueOffsets[task_id * level_count + thread_id]);
}

template<class WorkItem, std::size_t level_count>
__global__ void kernelMoveElements(WorkItem* work_stacks_in, WorkItem* work_stacks_out,
                                   uint32_t* work_stacks_item_count, uint32_t* item_sums_per_level,
                                   uint32_t* inter_stack_offsets, uint32_t* num_total_work_items, const uint32_t stack_size_per_task)
{
  __shared__ uint32_t shared_offsets[level_count * 2];

  const uint32_t task_id = blockIdx.x;
  const uint32_t thread_id = threadIdx.x;
  const uint32_t num_tasks = gridDim.x;
  const uint32_t num_threads = blockDim.x;
  const uint32_t num_elements = work_stacks_item_count[task_id];
  const WorkItem* my_work_stack = &work_stacks_in[task_id * stack_size_per_task];

// copy offsets into shared memory
#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    shared_offsets[i] = item_sums_per_level[(level_count - 1 - i) * num_tasks + task_id];

#pragma unroll
  for (uint32_t i = thread_id; i < level_count; i += num_threads)
    shared_offsets[level_count + i] = inter_stack_offsets[task_id * level_count + i];
  __syncthreads();

  // move elements
  for (uint32_t i = thread_id; i < num_elements; i += num_threads)
  {
    const uint32_t level = my_work_stack[i].level;
    const uint32_t offset = shared_offsets[level] + (i - shared_offsets[level_count + level]);
    assert(level < level_count);
    assert(i >= shared_offsets[level_count + level]);
    work_stacks_out[(offset % num_tasks) * stack_size_per_task + offset / num_tasks] = my_work_stack[i];
  }

  if (thread_id == 0)
  {
    uint32_t totalNum = item_sums_per_level[num_tasks * level_count];
    work_stacks_item_count[task_id] = totalNum / num_tasks + ((totalNum % num_tasks) >= (task_id + 1));
    if (task_id == 0)
      *num_total_work_items = totalNum;
  }
}

template<std::size_t num_threads, typename WorkItem, std::size_t branching_factor, std::size_t level_count>
void balanceWorkStacks(WorkItem* dev_work_stacks_in, WorkItem* dev_work_stacks_out,
                       uint32_t* dev_work_stacks_item_count, const uint32_t num_tasks,
                       uint32_t* host_num_total_work_items, const uint32_t stack_size_per_task)
{
  timespec time1 = getCPUTime();

  uint32_t* sums_per_level = NULL;
  uint32_t* inter_stack_offsets = NULL;
  uint32_t* dev_num_total_work_items = NULL;
  HANDLE_CUDA_ERROR(cudaMalloc(&inter_stack_offsets, sizeof(uint32_t) * level_count * num_tasks));
  HANDLE_CUDA_ERROR(cudaMalloc(&sums_per_level, sizeof(uint32_t) * (num_tasks * level_count + 1)));
  HANDLE_CUDA_ERROR(cudaMalloc(&dev_num_total_work_items, sizeof(uint32_t)));
  HANDLE_CUDA_ERROR(cudaMemset(&sums_per_level[num_tasks * level_count], 0, sizeof(uint32_t)));

  time1 = getCPUTime();
  kernelCountElements<WorkItem, level_count> <<<num_tasks, num_threads>>>(dev_work_stacks_in,
                                                                          dev_work_stacks_item_count,
                                                                          stack_size_per_task,
                                                                          sums_per_level,
                                                                          inter_stack_offsets);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  //printf("kernelCountElements: %f ms\n", timeDiff(time1, getCPUTime()));

  time1 = getCPUTime();

  // make prefix sum for each sumArray entry since the thrust function
  // can't be used to reduce a variable sized array as element
  thrust::exclusive_scan(thrust::device_ptr<uint32_t>(sums_per_level),
                         thrust::device_ptr<uint32_t>(sums_per_level + num_tasks * level_count + 1),
                         thrust::device_ptr<uint32_t>(sums_per_level));
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); // sync just like for plain kernel calls
  //printf("thrust::exclusive_scan: %f ms\n", timeDiff(time1, getCPUTime()));

  time1 = getCPUTime();
  // distribute data into new stacks
  kernelMoveElements<WorkItem, level_count> <<<num_tasks, num_threads>>>(dev_work_stacks_in,
                                                                         dev_work_stacks_out,
                                                                         dev_work_stacks_item_count,
                                                                         sums_per_level,
                                                                         inter_stack_offsets,
                                                                         dev_num_total_work_items,
                                                                         stack_size_per_task);

  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
  //printf("kernelMoveElements: %f ms\n", timeDiff(time1, getCPUTime()));

  HANDLE_CUDA_ERROR(
      cudaMemcpy(host_num_total_work_items, dev_num_total_work_items, sizeof(uint32_t),
                 cudaMemcpyDeviceToHost));

  time1 = getCPUTime();
  HANDLE_CUDA_ERROR(cudaFree(sums_per_level));
  HANDLE_CUDA_ERROR(cudaFree(inter_stack_offsets));
  HANDLE_CUDA_ERROR(cudaFree(dev_num_total_work_items));
}

// #########################################################################################
// ###################### Concept of kernel for processing the work ########################
// #########################################################################################

static __device__ __forceinline__
bool handleIdleCounter(const uint32_t num_stack_work_items, const uint32_t stack_items_threshold,
                       uint32_t* tasks_idle_counter, const uint32_t idle_count_threshold,
                       const uint32_t thread_id)
{
  if ((num_stack_work_items == 0) || (num_stack_work_items >= stack_items_threshold))
  {
    if (thread_id == 0)
      atomicInc(tasks_idle_counter, UINT_MAX);
    return true;
  }
  else
  {
    const bool idle_threshold_reached = __syncthreads_or(
        thread_id == 0 && (*tasks_idle_counter >= idle_count_threshold));
    return idle_threshold_reached;
  }
}

template<class _WorkItem, std::size_t num_threads, std::size_t branching_factor>
struct AbstractKernelConfig
{
public:

  // ###### Template parameter forwarding ######
  enum
  {
    CACHE_SIZE = num_threads / branching_factor,
    NUM_THREADS = num_threads,
    BRANCHING_FACTOR = branching_factor
  };
  typedef _WorkItem WorkItem;
  // ###########################################

  struct AbstractSharedMemConfig
  {
  public:
    uint32_t num_stack_work_items; // number of work items in stack
    WorkItem work_item_cache[CACHE_SIZE];
    WorkItem* my_work_stack; // pointer to stack of this task
  };

  struct AbstractSharedVolatileMemConfig
  {
  public:
    // nothing
  };

  struct AbstractVariablesConfig
  {
  public:
    uint32_t num_work_items;
    bool is_active;
  };

  struct AbstractConstConfig
  {
  public:
    const dim3 grid_dim;
    const dim3 block_dim;
    const dim3 block_ids;
    const dim3 thread_ids;

    const uint32_t block_id;
    const uint32_t thread_id;
    const uint32_t warp_id;
    const uint32_t warp_lane;
    const uint32_t work_index;
    const uint32_t work_lane;

    const uint32_t stack_items_threshold;

    __host__ __device__
    AbstractConstConfig(const dim3 p_grid_dim,
                        const dim3 p_block_dim,
                        const dim3 p_block_ids,
                        const dim3 p_thread_ids,
                        const uint32_t p_stack_size_per_task,
                        const uint32_t p_stack_items_threshold) :
        grid_dim(p_grid_dim),
        block_dim(p_block_dim),
        block_ids(p_block_ids),
        thread_ids(p_thread_ids),
        block_id(p_block_ids.x),
        thread_id(p_thread_ids.x),
        warp_id(thread_id / WARP_SIZE),
        warp_lane(thread_id % WARP_SIZE),
        work_index(thread_id / branching_factor),
        work_lane(thread_id % branching_factor),
        stack_items_threshold(p_stack_items_threshold)
    {

    }
  };

  struct AbstractKernelParameters
  {
  public:
    WorkItem* work_stacks;
    uint32_t* work_stacks_item_count;
    const uint32_t stack_size_per_task;
    uint32_t* tasks_idle_count;
    const uint32_t idle_count_threshold;

    __host__ __device__
    AbstractKernelParameters(WorkItem* p_work_stacks, uint32_t* p_work_stacks_item_count,
                             const uint32_t p_stack_size_per_task, uint32_t* p_tasks_idle_count,
                             const uint32_t p_idle_count_threshold) :
        work_stacks(p_work_stacks),
        work_stacks_item_count(p_work_stacks_item_count),
        stack_size_per_task(p_stack_size_per_task),
        tasks_idle_count(p_tasks_idle_count),
        idle_count_threshold(p_idle_count_threshold)
    {

    }

    __host__ __device__
    AbstractKernelParameters(const AbstractKernelParameters& params) :
        work_stacks(params.work_stacks),
        work_stacks_item_count(params.work_stacks_item_count),
        stack_size_per_task(params.stack_size_per_task),
        tasks_idle_count(params.tasks_idle_count),
        idle_count_threshold(params.idle_count_threshold)
    {

    }
  };

  typedef AbstractSharedMemConfig SharedMem;
  typedef AbstractSharedVolatileMemConfig SharedVolatileMem;
  typedef AbstractVariablesConfig Variables;
  typedef AbstractConstConfig Constants;
  typedef AbstractKernelParameters KernelParams;

  __device__
  static void doLoadBalancedWork(SharedMem& shared_mem, volatile SharedVolatileMem& shared_volatile_mem,
                                 Variables& variables, const Constants& constants, KernelParams& kernel_params);

  __device__
  static void doReductionWork(SharedMem& shared_mem, volatile SharedVolatileMem& shared_volatile_mem,
                              Variables& variables, const Constants& constants, KernelParams& kernel_params);

  __device__
  static bool abortLoop(SharedMem& shared_mem, volatile SharedVolatileMem& shared_volatile_mem,
                              Variables& variables, const Constants& constants, KernelParams& kernel_params)
  {
    if ((shared_mem.num_stack_work_items == 0) || (shared_mem.num_stack_work_items >= constants.stack_items_threshold))
    {
      if (constants.thread_id == 0)
        atomicInc(kernel_params.tasks_idle_count, UINT_MAX);
      return true;
    }
    else
    {
      const bool idle_threshold_reached = __syncthreads_or(
          constants.thread_id == 0 && (*kernel_params.tasks_idle_count >= kernel_params.idle_count_threshold));
      return idle_threshold_reached;
    }
  }
};

template<class LBKernelConfig>
__global__
void kernelLBWorkConcept(typename LBKernelConfig::KernelParams kernel_params)
{
  __shared__ typename LBKernelConfig::SharedMem shared_mem;
  volatile __shared__ typename LBKernelConfig::SharedVolatileMem shared_volatile_mem;
  typename LBKernelConfig::Variables variables;
  const typename LBKernelConfig::Constants constants(gridDim, blockDim, blockIdx, threadIdx, kernel_params.stack_size_per_task);
  __syncthreads(); // make sure race conditions for initializing the shared memory doen't lead to data inconsistency

  if (constants.thread_id == 0)
  {
    shared_mem.num_stack_work_items = kernel_params.work_stacks_item_count[constants.block_id];
    shared_mem.my_work_stack = &kernel_params.work_stacks[constants.block_id * kernel_params.stack_size_per_task];
  }
  __syncthreads();

  assert(shared_mem.num_stack_work_items < constants.stack_items_threshold);

  while (true)
  {
    //    if (handleIdleCounter(shared_mem.num_stack_work_items, constants.stack_items_threshold, kernel_params.tasks_idle_count,
    //                          kernel_params.idle_count_threshold, constants.thread_id))
    if(LBKernelConfig::abortLoop(shared_mem, shared_volatile_mem, variables, constants, kernel_params))
      break;

    variables.num_work_items = min((uint32_t) (LBKernelConfig::NUM_THREADS / LBKernelConfig::BRANCHING_FACTOR), shared_mem.num_stack_work_items);
    variables.is_active = constants.work_index < variables.num_work_items;

    // every thread grabs some work
    blockCopy(shared_mem.work_item_cache, &shared_mem.my_work_stack[shared_mem.num_stack_work_items - variables.num_work_items],
              variables.num_work_items * sizeof(LBKernelConfig::WorkItem), constants.thread_id, LBKernelConfig::NUM_THREADS);
    __syncthreads();

    // decrease num work items in stack by the grabbed work
    if (constants.thread_id == 0)
      shared_mem.num_stack_work_items -= variables.num_work_items;
    __syncthreads();

    LBKernelConfig::doLoadBalancedWork(shared_mem, shared_volatile_mem, variables, constants, kernel_params);
  }

  LBKernelConfig::doReductionWork(shared_mem, shared_volatile_mem, variables, constants, kernel_params);

  if (constants.thread_id == 0)
    kernel_params.work_stacks_item_count[constants.block_id] = shared_mem.num_stack_work_items;
}

}
}
}

#endif
