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
#ifndef GPU_VOXELS_BIT_VECTOR_H_INCLUDED
#define GPU_VOXELS_BIT_VECTOR_H_INCLUDED

#include <cstring>
#include <cuda.h>

namespace gpu_voxels {

/**
 * @brief This template class represents a vector of bits with a given length.
 */
template<std::size_t length>
class BitVector
{
public:
  typedef uint8_t item_type;

  __host__     __device__
  BitVector();

  /**
   * @brief clear Sets all bits to zero.
   */
  __host__ __device__
  void clear();

  /**
   * @brief operator | Bitwise or-operator
   * @param o Other operand

   */
  __host__     __device__
  BitVector<length> operator|(const BitVector<length>& o) const;

  /**
   * @brief operator |= Bitwise or-operator
   * @param o Other operand
   */
  __host__ __device__
  void operator|=(const BitVector<length>& o);

  /**
   * @brief operator ~ Bitwise not-operator
   * @return Returns the bitwise not of 'this'
   */
  __host__     __device__
  BitVector<length> operator~() const;

  /**
   * @brief operator ~ Bitwise and-operator
   * @return Returns the bitwise and of 'this'
   */
  __host__     __device__
  BitVector<length> operator&(const BitVector<length>& o) const;

  /**
   * @brief isZero Checks the bit vector for zero
   * @return True if all bits are zero, false otherwise
   */
  __host__ __device__
  bool isZero() const;

  /**
   * @brief getBit Gets the bit with at the given index.
   * @return Value of the selected bit.
   */
  __host__ __device__
  bool getBit(const uint32_t index) const;

  /**
   * @brief clearBit Clears the bit at the given index
   */
  __host__ __device__
  void clearBit(const uint32_t index);

  /**
   * @brief setBit Sets the bit with at the given index.
   */
  __host__ __device__
  void setBit(const uint32_t index);

  /**
   * @brief getByte Gets the byte to the given index position.
   * @return Byte of given index position
   */
  __host__ __device__
  item_type getByte(const uint32_t index) const;

  /**
   * @brief setByte Sets the byte at the given index position.
   * @param index Which byte to set
   * @param data Data to write into byte
   */
  __host__ __device__
  void setByte(const uint32_t index, const item_type data);

protected:

  /**
   * @brief getBit Gets the reference to the given index position.
   * @return Reference to byte of given index position
   */
  __host__ __device__
  item_type* getByte(const uint32_t index);

protected:
  static const uint32_t m_size = (length + 7) / 8;
  item_type m_bytes[m_size];
};

} // end of ns

#endif
