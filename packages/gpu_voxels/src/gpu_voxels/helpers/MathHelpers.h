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
#ifndef GPU_VOXELS_HELPERS_MATH_HELPERS_H_INCLUDED
#define GPU_VOXELS_HELPERS_MATH_HELPERS_H_INCLUDED

#include <assert.h>
#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>

namespace gpu_voxels {

/*!
 * \brief computeLinearLoad
 * \param nr_of_items
 * \param blocks
 * \param threads_per_block
 */
void computeLinearLoad(const uint32_t nr_of_items, uint32_t* blocks, uint32_t* threads_per_block);

/*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
float interpolateLinear(float value1, float value2, float ratio);

/*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
double interpolateLinear(double value1, double value2, double ratio);

/*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
 *  using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
std::vector<float> interpolateLinear(const std::vector<float>& joint_state1,
                                     const std::vector<float>& joint_state2, float ratio);

/*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
 *  using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
std::vector<double> interpolateLinear(const std::vector<double>& joint_state1,
                                      const std::vector<double>& joint_state2, double ratio);

/*! Interpolate linear between the robot JointValueMaps \a joint_state1 and \a joint_state2
 *  using the given \a ratio.
 *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
 *  middle.
 */
std::map<std::string, float> interpolateLinear(const std::map<std::string, float>& joint_state1,
                                               const std::map<std::string, float>& joint_state2, float ratio);
} // end of namespace

#endif
