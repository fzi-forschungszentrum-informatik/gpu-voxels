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
 * \date    2013-11-07
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_DATATYPES_H_INCLUDED
#define GPU_VOXELS_OCTREE_DATATYPES_H_INCLUDED

//#define DEBUG_MODE

//#include <cstdio>
//#include <iostream>
#include <stdint.h> // for fixed size datatypes
#include <assert.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <sys/utsname.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/BitVector.h>

namespace gpu_voxels {
namespace NTree {

//-Wno-unknown-pragmas -Wno-unused-function

//#define PROBABILISTIC_TREE
#define PACKING_OF_VOXEL true
#define DO_REBUILDS
#define PERFORMANCE_MEASUREMENT

#define VISUALIZER_SHIFT_X 0//7600
#define VISUALIZER_SHIFT_Y 0//7600
#define VISUALIZER_SHIFT_Z 0//8000
// only include declaration of template class NTree.hpp once to speed up the build process
#define NTREE_PRECOMPILE

#undef TREAT_UNKNOWN_AS_COLLISION // When defined, intesections with unknown octree nodes will result in collisions

#define INITIAL_PROBABILITY Probability(0)  // probability used to initialize any new node
#define INITIAL_FREE_SPACE_PROBABILITY Probability(0)
#define INITIAL_OCCUPIED_PROBABILITY Probability(0)
#define FREE_UPDATE_PROBABILITY Probability(-10)
#define OCCUPIED_UPDATE_PROBABILITY Probability(100)

#define KINECT_CUT_FREE_SPACE_Y 0//50
#define KINECT_CUT_FREE_SPACE_X 0//80
#define KINECT_FREE_NAN_MEASURES true
//#define VISUALIZER_OBJECT_DATA_ONLY
//#define KINECT_FREE_SPACE_DEBUG
//#define KINECT_FREE_SPACE_DEBUG_2
#define KINECT_WIDTH 640
#define KINECT_HEIGHT 480

//#define KINECT_ORIENTATION Vector3f(M_PI / 2.0f, M_PI, 0) // up-side-down on PTU (roll, pitch yaw)
#define KINECT_ORIENTATION Vector3f(M_PI / 2.0f, 0, 0) // normal (roll, pitch yaw)

#define NUM_BLOCKS 2688 // 8192 * 8;
#define NUM_THREADS_PER_BLOCK 128 // 32 // 32 * 8
#define THRESHOLD_OCCUPANCY 10
#define D_PTR(X) thrust::raw_pointer_cast((X).data())
#define MAX_VALUE(TYPE) ((TYPE)((1 << (sizeof(TYPE) * 8)) - 1))

#define INVALID_VOXEL ULONG_MAX
const gpu_voxels::Vector3ui INVALID_POINT = gpu_voxels::Vector3ui(UINT_MAX, UINT_MAX, UINT_MAX);

#define DISABLE_SEPARATE_COMPILTION

#define VOXELMAP_FLAG_SIZE 1

// NTree
#define FEW_MESSAGES
//#define PROPAGATE_MESSAGES
//#define INSERT_MESSAGES
//#define FREESPACE_MESSAGES
#define REBUILD_MESSAGES
//#define EXTRACTCUBE_MESSAGES
//#define TRAFO_MESSAGES
//#define NEWKINECTDATA_MESSAGES
#define INTERSECT_MESSAGES
//#define FREE_BOUNDING_BOX_MESSAGES
//#define SENSOR_DATA_PREPROCESSING_MESSAGES

#define KINECT_PREPROCESSS_ON_GPU
//#define CHECK_SORTING
#define LOAD_BALANCING_PROPAGATE
#define PROPAGATE_BOTTOM_UP
//#define COUNT_BEFORE_EXTRACT

// Kinect
//#define DEPTHCALLBACK_MESSAGES

#define MODE_KINECT
//#define MANUAL_MODE
//#define PAN_TILT
//#define IGNORE_FREE_SPACE

// ########## LoadBalancer ##########
#define DEFAULT_PROPAGATE_QUEUE_NTASKS 1024 // good choice for number of tasks/blocks due to experimental evaluation
#define PROPAGATE_TRAVERSAL_THREADS 128//128
#define PROPAGATE_IDLETASKS_FOR_ABORT_FACTOR 2/3

#define QUEUE_NTASKS 2688//2688//96//1024 //128
#define QUEUE_SIZE_PER_TASK 2600//2600
#define QUEUE_SIZE_PER_TASK_GLOBAL 700//50000
#define QUEUE_IDLETASKS_FOR_ABORT QUEUE_NTASKS*2/3//2/3
#define TRAVERSAL_THREADS 128 //32//64 //64
#define QUEUE_SIZE_PER_TASK_INIT TRAVERSAL_THREADS*3
#define WARP_SIZE 32
#define MAX_NUMBER_OF_THREADS 1024

// Define min/max functions to handle different namespaces of host and device code
#undef MIN
#undef MAX
#ifdef __CUDACC__
#define MIN(x,y) min(x,y)
#define MAX(x,y) max(x,y)
#else
#define MIN(x,y) std::min(x,y)
#define MAX(x,y) std::max(x,y)
#endif

// ######################################################################

typedef uint32_t voxel_count;

/*
 * Returns the difference in milliseconds
 */
inline static double timeDiff(timespec start, timespec end)
{
  double ms = 0.0;
  if ((end.tv_nsec - start.tv_nsec) < 0)
  {
    ms = double(1000000000 + end.tv_nsec - start.tv_nsec) / 1000000.0;
    ms += double(end.tv_sec - start.tv_sec - 1) * 1000.0;
  }
  else
  {
    ms = double(end.tv_nsec - start.tv_nsec) / 1000000.0;
    ms += double(end.tv_sec - start.tv_sec) * 1000.0;
  }
  return ms;
}

inline static timespec getCPUTime()
{
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t;
}

inline static std::string getTime_str()
{
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time(&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer, 80, "%F_%H.%M.%S", timeinfo);
  return buffer;
}

inline std::string to_string(int _Val, const char format[] = "%d")
{   // convert long long to string
  char _Buf[50];
  sprintf(_Buf, format, _Val);
  return (std::string(_Buf));
}

inline utsname getUname()
{
  utsname tmp;
  uname(&tmp);
  return tmp;
}

//#define lookup_type_8 uint8_t
//#define lookup_type_64 uint8_t
//#define lookup_type_512 uint16_t
//#define lookup_type_4096 uint16_t
//#define lookup_type(X) lookup_type_"X"

//#define third_root(X) third_root_"X"
//#define third_root_8 2
//#define third_root_64 4
//#define third_root_512 8

//// ##### type comparison at compile time #####
//template<typename T>
//struct is_same<T, T>
//{
//    static const bool value = true;
//};
//
//template<typename T, typename U>
//struct is_same
//{
//    static const bool value = false;
//};
//
//template<typename T, typename U>
//bool eqlTypes() { return is_same<T, U>::value; }
//// #################################################

__host__ inline
uint32_t linearApprox(const float y1, const float x1, const float y2, const float x2, const float x,
                      const uint32_t alignment = 1, const uint32_t max_val = UINT_MAX)
{
    const float a = (y1 - y2) / (x1 - x2);
    const float b = y1 - a * x1;
    const float y = a * x + b;
    const float y_aligned = ceil(y / alignment) * alignment;
    const uint32_t y_min_max = std::min(uint32_t(std::max(y_aligned, float(alignment))), max_val);

    return y_min_max;
}

} // end of ns
} // end of ns
#endif
