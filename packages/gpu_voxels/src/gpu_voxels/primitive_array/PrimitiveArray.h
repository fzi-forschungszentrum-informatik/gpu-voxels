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
 * \date    2014-12-15
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_PRIMITIVE_ARRAY_PRIMITIVE_ARRAY_H_INCLUDED
#define GPU_VOXELS_PRIMITIVE_ARRAY_PRIMITIVE_ARRAY_H_INCLUDED

#include <gpu_voxels/helpers/cuda_datatypes.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/logging/logging_primitive_array.h>

#include <vector>
#include <string>

/**
 * @namespace gpu_voxels::primitive_array
 * Contains implementation of PrimitiveArray Datastructure and according operations
 */
namespace gpu_voxels {
namespace primitive_array {

enum PrimitiveType
{
  ePRIM_SPHERE = 0,
  ePRIM_CUBOID = 1,
  ePRIM_INITIAL_VALUE = 255 // initial value to determine if the type has changed.
};


class PrimitiveArray;
typedef boost::shared_ptr<PrimitiveArray> PrimitiveArraySharedPtr;

class PrimitiveArray
{
public:

  PrimitiveArray(const Vector3ui dim, const float voxel_side_length, PrimitiveType type);
  ~PrimitiveArray();

  //! updates the points in the array
  void setPoints(const std::vector<Vector4f> &points);
  void setPoints(const std::vector<Vector4i> &points);

  //! generate entities at points, all with same diameter
  void setPoints(const std::vector<Vector3f> &points, const float &diameter);
  void setPoints(const std::vector<Vector3i> &points, const uint32_t &diameter);

  //! get pointer to data array on device
  void* getVoidDeviceDataPtr() const { return (void*)m_dev_ptr_to_primitive_positions; }

  //! get the number of entities in the array
  uint32_t getNumPrimitives() const { return m_num_entities; }

  //! get the type of the entity
  PrimitiveType getPrimitiveType() const { return m_primitive_type; }

  //! get the diameter of the drawn entities.
  float getDiameter() const { return m_diameter; }

  //! get the number of bytes that is required for the array
  virtual std::size_t getMemoryUsage();

private:
  Vector3ui m_dim;
  float m_voxel_side_length;
  Vector3ui getDimensions() const;
  Vector3f getMetricDimensions() const;

  void storePoints(const std::vector<Vector4f> &points);

  PrimitiveType m_primitive_type;
  uint32_t m_num_entities;
  Vector4f* m_dev_ptr_to_primitive_positions;
  float m_diameter;
};

} // end of namespace primitive_array
} // end of namespace gpu_voxels
#endif
