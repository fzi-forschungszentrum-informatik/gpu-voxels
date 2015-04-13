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
 * \date    2014-07-08
 *
 */
//----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_BIT_VECTOR_HPP_INCLUDED
#define GPU_VOXELS_BIT_VECTOR_HPP_INCLUDED

#include <gpu_voxels/helpers/BitVector.h>
#include <gpu_voxels/helpers/cuda_handling.h>

namespace gpu_voxels {

template<std::size_t length>
__host__ __device__
BitVector<length>::BitVector()
{
  clear();
}

template<std::size_t length>
__host__ __device__
void BitVector<length>::clear()
{
  memset(m_bytes, 0, sizeof(m_bytes));
}

template<std::size_t length>
__host__     __device__
BitVector<length> BitVector<length>::operator|(const BitVector<length>& o) const
{
  BitVector<length> res;
#ifdef  __CUDA_ARCH__
#pragma unroll
#endif
  for (uint32_t i = 0; i < m_size; ++i)
    res.m_bytes[i] = m_bytes[i] | o.m_bytes[i];
  return res;
}

template<std::size_t length>
__host__ __device__
void BitVector<length>::operator|=(const BitVector<length>& o)
{
#ifdef  __CUDA_ARCH__
#pragma unroll
#endif
  for (uint32_t i = 0; i < m_size; ++i)
    m_bytes[i] |= o.m_bytes[i];
}

template<std::size_t length>
__host__     __device__
BitVector<length> BitVector<length>::operator~() const
{
  BitVector<length> res;
#ifdef  __CUDA_ARCH__
#pragma unroll
#endif
  for (uint32_t i = 0; i < m_size; ++i)
    res.m_bytes[i] = ~res.m_bytes[i];
  return res;
}

template<std::size_t length>
__host__     __device__
BitVector<length> BitVector<length>::operator&(const BitVector<length>& o) const
{
  BitVector<length> res;
#ifdef  __CUDA_ARCH__
#pragma unroll
#endif
  for (uint32_t i = 0; i < m_size; ++i)
    res.m_bytes[i] = res.m_bytes[i] & o.m_bytes[i];
  return res;
}

template<std::size_t length>
__host__ __device__
bool BitVector<length>::isZero() const
{
  bool result = true;
#ifdef  __CUDA_ARCH__
#pragma unroll
#endif
  for (uint32_t i = 0; i < m_size; ++i)
    result &= m_bytes[i] == 0;
  return result;
}

template<std::size_t length>
__host__     __device__
BitVector<length>::item_type* BitVector<length>::getByte(const uint32_t index)
{
  return &m_bytes[index >> 3];
}

template<std::size_t length>
__host__     __device__
BitVector<length>::item_type BitVector<length>::getByte(const uint32_t index) const
{
  return m_bytes[index >> 3];
}

template<std::size_t length>
__host__ __device__
bool BitVector<length>::getBit(const uint32_t index) const
{
  return getByte(index) & (1 << (index & 7));
}

template<std::size_t length>
__host__ __device__
void BitVector<length>::clearBit(const uint32_t index)
{
  item_type* selected_byte = getByte(index);
  *selected_byte = *selected_byte & item_type(~(1 << (index & 7)));
}

template<std::size_t length>
__host__ __device__
void BitVector<length>::setBit(const uint32_t index)
{
  item_type* selected_byte = getByte(index);
  *selected_byte = *selected_byte | item_type(1 << (index & 7));
}

template<std::size_t length>
__host__ __device__
void BitVector<length>::setByte(const uint32_t index, const item_type data)
{
  item_type* selected_byte = getByte(index);
  *selected_byte = data;
}

} // end of ns

#endif
