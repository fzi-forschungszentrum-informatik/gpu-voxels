// this is for emacs file handling -&- mode: c++; indent-tabs-mode: nil -&-

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
 * \date    2015-10-02
 *
 */
//----------------------------------------------------------------------

#include <assert.h>
#include <gpu_voxels/helpers/MathHelpers.h>

namespace gpu_voxels {

void computeLinearLoad(const uint32_t nr_of_items, uint32_t* blocks, uint32_t* threads_per_block)
{
//  if (nr_of_items <= cMAX_NR_OF_BLOCKS)
//  {
//    *blocks = nr_of_items;
//    *threads_per_block = 1;
//  }
//  else
//  {

  if(nr_of_items == 0)
  {
      LOGGING_WARNING(
          Gpu_voxels_helpers,
          "Number of Items is 0. Blocks and Threads per Block is set to 1. Size 0 would lead to a Cuda ERROR" << endl);

      *blocks = 1;
      *threads_per_block = 1;
      return;
  }

  if (nr_of_items <= cMAX_NR_OF_BLOCKS * cMAX_THREADS_PER_BLOCK)
  {
    *blocks = (nr_of_items + cMAX_THREADS_PER_BLOCK - 1)
            / cMAX_THREADS_PER_BLOCK;                          // calculation replaces a ceil() function
    *threads_per_block = cMAX_THREADS_PER_BLOCK;
  }
  else
  {
    /* In this case the kernel must perform multiple runs because
     * nr_of_items is larger than the gpu can handle at once.
     * To overcome this limit, use standard parallelism offsets
     * as when programming host code (increment by the number of all threads
     * running). Use something like
     *
     *   uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
     *
     *   while (i < nr_of_items)
     *   {
     *     // perform some kernel operations here
     *
     *     // increment by number of all threads that are running
     *     i += blockDim.x * gridDim.x;
     *   }
     *
     * CAUTION: currently cMAX_NR_OF_BLOCKS is 64K, although
     *          GPUs with SM >= 3.0 support up to 2^31 -1 blocks in a grid!
     */
    LOGGING_ERROR(
      Gpu_voxels_helpers,
      "computeLinearLoad: Number of Items " << nr_of_items << " exceeds the limit cMAX_NR_OF_BLOCKS * cMAX_THREADS_PER_BLOCK = " << (cMAX_NR_OF_BLOCKS*cMAX_THREADS_PER_BLOCK) << "! This number of items cannot be processed in a single invocation." << endl);
    *blocks = cMAX_NR_OF_BLOCKS;
    *threads_per_block = cMAX_THREADS_PER_BLOCK;
  }
}


/*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
float interpolateLinear(float value1, float value2, float ratio)
{
  return (value1 * (1.0 - ratio) + value2 * ratio);
}

/*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
double interpolateLinear(double value1, double value2, double ratio)
{
  return (value1 * (1.0 - ratio) + value2 * ratio);
}

/*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
 *  using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
std::vector<float> interpolateLinear(const std::vector<float>& joint_state1,
                                     const std::vector<float>& joint_state2, float ratio)
{
  assert(joint_state1.size() == joint_state2.size());

  std::vector<float> result(joint_state1.size());
  for (std::size_t i=0; i<joint_state1.size(); ++i)
  {
    result.at(i) = interpolateLinear(joint_state1.at(i), joint_state2.at(i), ratio);

  }
  return result;
}

/*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
 *  using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
std::vector<double> interpolateLinear(const std::vector<double>& joint_state1,
                                      const std::vector<double>& joint_state2, double ratio)
{
  assert(joint_state1.size() == joint_state2.size());

  std::vector<double> result(joint_state1.size());
  for (std::size_t i=0; i<joint_state1.size(); ++i)
  {
    result.at(i) = interpolateLinear(joint_state1.at(i), joint_state2.at(i), ratio);

  }
  return result;
}

/*! Interpolate linear between the robot JointValueMaps \a joint_state1 and \a joint_state2
 *  using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
std::map<std::string, float> interpolateLinear(const std::map<std::string, float>& joint_state1,
                                               const std::map<std::string, float>& joint_state2, float ratio)
{
  assert(joint_state1.size() == joint_state2.size());

  std::map<std::string, float> result(joint_state1);
  for (std::map<std::string, float>::const_iterator it=joint_state1.begin();
       it!=joint_state1.end(); ++it)
  {
    result[it->first] = interpolateLinear(joint_state1.at(it->first),
                                          joint_state2.at(it->first), ratio);
  }
  return result;
}

} // end of namespace
