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
 * \date    2013-11-16
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_SENSORMODEL_H_INCLUDED
#define GPU_VOXELS_OCTREE_SENSORMODEL_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/octree/Voxel.h>
#include <algorithm>


namespace gpu_voxels {
namespace NTree {

/**
 * @brief Sensor model for handling the probabilistic sensor data and estimate probabilities after transforming the point cloud into voxel representation.
 */
struct SensorModel
{
private:
  Probability update_probabilty;
  Probability initial_probability;

public:

  __device__ __host__
  SensorModel()
  {
    update_probabilty = initial_probability = 0;
  }

  /**
   * @brief Constructs a new SensorModel with the given initial and update probabilities.
   * @param update_probabilty Probability applied if the same sensor reading is the same.
   * @param initial_probability Probability applied for the first sensor reading of this kind.
   */
  __device__ __host__
  SensorModel(const Probability update_probabilty, const Probability initial_probability)
  {
    this->update_probabilty = update_probabilty;
    this->initial_probability = initial_probability;
  }

  /**
   * @brief Applies the sensor model for the given sensor reading.
   * @param point
   * @return
   */
  __device__ __host__ __forceinline__
  Probability applySensorModel(const gpu_voxels::Vector3f point) const
  {
    return initial_probability;
  }

  /**
   * @brief Estimates the probability of a super-voxel consisting of the given sequence of voxels.
   * @param first
   * @param last
   * @return
   */
  __device__ __host__ __forceinline__
  Probability estimateVoxelProbability(Voxel* first, Voxel* last) const
  {
    Probability occ = first->getOccupancy();
    for (Voxel* i = first + 1; i < last; ++i)
      // watch out for overflow: cast to int32_t
      occ = MIN(MAX( int32_t(int32_t(occ) + int32_t(i->getOccupancy())), int32_t(MIN_PROBABILITY)), int32_t(MAX_PROBABILITY));
    return occ;
  }

  /**
   * @brief Estimates the probability of a voxel, by the number of sensor points, which lie in it.
   * @param count Number of sensor points rediding inside this voxel.
   * @return
   */
  __device__ __host__ __forceinline__
  Probability estimateVoxelProbability(const voxel_count count) const
  {
    // watch out for overflow: cast to int32_t
    return MIN(MAX( int32_t(int32_t(initial_probability) + int32_t(update_probabilty) * count), int32_t(MIN_PROBABILITY)), int32_t(MAX_PROBABILITY));
  }

  /**
   * @brief Estimates the probability based on the last estimation and an observation. See Bayesian filter!
   * @param lastProbEstimation
   * @param observedProb
   * @return
   */
  __device__ __host__ __forceinline__
  Probability estimateProbability(const Probability lastProbEstimation, const Probability observedProb) const
  {
    // TODO it's just a stub
    return (lastProbEstimation + observedProb) / 2;
  }

  /**
   * @brief getInitialProbability
   * @return Probability applied for the first sensor reading of this kind.
   */
  __device__ __host__ __forceinline__
  Probability getInitialProbability() const
  {
    return initial_probability;
  }

  /**
   * @brief getUpdateProbability
   * @return Probability applied if the same sensor reading is the same.
   */
  __device__ __host__ __forceinline__
  Probability getUpdateProbability() const
  {
    return update_probabilty;
  }

  /**
   * @brief setInitialProbability
   * @param initial_probability Probability applied for the first sensor reading of this kind.
   */
  __device__ __host__ __forceinline__
  void setInitialProbability(const Probability initial_probability)
  {
    this->initial_probability = initial_probability;
  }

  /**
   * @brief setUpdateProbability
   * @param update_probabilty Probability applied if the same sensor reading is the same.
   */
  __device__ __host__ __forceinline__
  void setUpdateProbability(const Probability update_probabilty)
  {
    this->update_probabilty = update_probabilty;
  }
};

}  // end of ns
}  // end of ns
#endif /* SENSORMODEL_H_ */
