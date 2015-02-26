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
 * \date    2015-01-05
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/primitive_array/PrimitiveArray.h>
#include <gpu_voxels/helpers/cuda_handling.hpp>

namespace gpu_voxels {
namespace primitive_array {

PrimitiveArray::PrimitiveArray(PrimitiveType type) :
  m_primitive_type(type),
  m_num_entities(0),
  m_dev_ptr_to_primitive_positions(NULL)
{
}

PrimitiveArray::~PrimitiveArray()
{
  if(m_dev_ptr_to_primitive_positions)
  {
    //delete old array
    cudaFree(m_dev_ptr_to_primitive_positions);
    m_dev_ptr_to_primitive_positions = NULL;
  }
}

void PrimitiveArray::setPoints(const std::vector<Vector4f> &points)
{
  if(m_dev_ptr_to_primitive_positions)
  {
    //delete old array
    cudaFree(m_dev_ptr_to_primitive_positions);
    m_dev_ptr_to_primitive_positions = NULL;
  }
  m_num_entities = points.size();
  // allocate the accumulated memory for the positions of the primitives
  HANDLE_CUDA_ERROR(
      cudaMalloc((void** )&m_dev_ptr_to_primitive_positions, m_num_entities * sizeof(Vector4f)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(m_dev_ptr_to_primitive_positions, points.data(), m_num_entities * sizeof(Vector4f),
                 cudaMemcpyHostToDevice));
}


uint32_t PrimitiveArray::getMemorySizeInByte()
{
  return m_num_entities * sizeof(Vector4f);
}

} // end of namespace
} // end of namespace
