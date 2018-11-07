// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2009-06-13
 *
 * \brief   Contains icl_core::tRingBuffer
 *
 * \b icl_core::tRingBuffer
 *
 * A simple ring buffer implementation based on tVector.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_T_RING_BUFFER_H_INCLUDED
#define ICL_CORE_T_RING_BUFFER_H_INCLUDED

#include <stdexcept>
#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/Deprecate.h"

namespace icl_core {

/*! \brief A simple ring buffer implementation based on std::vector.
 */
template <typename T>
class ICL_CORE_VC_DEPRECATE tRingBuffer
{
public:
  typedef T value_type;
  typedef size_t size_type;
  static const size_type cDEFAULT_CAPACITY = 32;

  //! Default constructor.
  tRingBuffer(size_type capacity = cDEFAULT_CAPACITY)
    : m_buffer(capacity+1), m_write(0), m_read(0)
  { }

  //! Copy constructor.
  tRingBuffer(const tRingBuffer<T>& other)
    : m_buffer(other.m_buffer), m_write(other.m_write), m_read(other.m_read)
  { }

  //! Assignment operator.
  tRingBuffer& operator = (const tRingBuffer<T>& other)
  {
    m_buffer = other.m_buffer;
    m_write = other.m_write;
    m_read = other.m_read;
    return *this;
  }

  //! Clears the ring buffer.
  void Clear() { m_write = m_read = 0; }

  //! Adds an element to the ring buffer provided there is room.
  /*! If \a overwrite == \c false, throws an exception if the element
   *  can not be added.  If \a overwrite == \c true, old elements are
   *  discarded to make room for new ones.
   */
  void Write(const T& val, bool overwrite=false)
  {
    size_type new_write_pos = m_write+1;
    if (new_write_pos >= m_buffer.size())
    {
      new_write_pos = 0;
    }
    if (new_write_pos == m_read)
    {
      if (overwrite)
      {
        Skip();
      }
      else
      {
        throw std::out_of_range("tRingBuffer::Write: capacity exceeded");
      }
    }
    m_buffer[m_write] = val;
    m_write = new_write_pos;
  }

  //! Read an arbitrary element from the ring buffer without removing it.
  /*! Throws an exception if the index is out of range.
   *  \param pos The position into the buffer. 0 is the oldest element
   *  currently present.
   */
  const T& At(size_type pos) const
  {
    if (pos < Size())
    {
      pos += m_read;
      if (pos >= m_buffer.size())
      {
        pos -= m_buffer.size();
      }
      return m_buffer[pos];
    }
    else
    {
      throw std::out_of_range("tRingBuffer::Peek: out of range");
    }
  }

  //! Access an arbitrary element in the ring buffer without removing it.
  /*! Throws an exception if the index is out of range.
   *  \param pos The position into the buffer. 0 is the oldest element
   *  currently present.
   */
  T& At(size_type pos)
  {
    if (pos < Size())
    {
      pos += m_read;
      if (pos >= m_buffer.size())
      {
        pos -= m_buffer.size();
      }
      return m_buffer[pos];
    }
    else
    {
      throw std::out_of_range("tRingBuffer::Peek: out of range");
    }
  }

  //! Removes an element from the ring buffer without returning it.
  void Skip()
  {
    if (m_write == m_read)
    {
      throw std::out_of_range("tRingBuffer::Skip: buffer empty");
    }
    m_read++;
    if (m_read >= m_buffer.size())
    {
      m_read = 0;
    }
  }

  //! Removes an element from the ring buffer provided there is one present.
  /*! Throws an exception if the buffer is empty.
   */
  T Read()
  {
    if (m_write == m_read)
    {
      throw std::out_of_range("tRingBuffer::Read: buffer empty");
    }
    size_type read_pos = m_read;
    m_read++;
    if (m_read >= m_buffer.size())
    {
      m_read = 0;
    }
    return m_buffer[read_pos];
  }

  //! Returns the current number of elements in the ring buffer.
  inline size_type Size() const
  {
    if (m_write >= m_read)
    {
      return m_write-m_read;
    }
    else
    {
      return m_write+m_buffer.size()-m_read;
    }
  }

  //! Returns the capacity of the ring buffer.
  size_type Capacity() const { return m_buffer.size()-1; }

  //! Changes the capacity of the ring buffer.
  /*! If the new \a capacity is less than the current buffer size,
   *  only the latest \a capacity elements are kept, the rest are
   *  destroyed.
   */
  void SetCapacity(size_type capacity)
  {
    size_type old_size = Size();
    size_type new_size = (capacity < old_size) ? capacity : old_size;
    std::vector<T> old_buffer(m_buffer);
    size_type old_read = m_read;

    // Adjust capacity.
    m_buffer.resize(capacity+1);
    // Skip elements that will not fit.
    old_read += old_size-new_size;
    if (old_read >= old_buffer.size())
    {
      old_read -= old_buffer.size();
    }
    // Copy the rest.
    for (size_type i=0; i<new_size; i++)
    {
      m_buffer[i] = old_buffer[old_read];
      old_read++;
      if (old_read >= old_buffer.size())
      {
        old_read = 0;
      }
    }
    // Update pointers.
    m_read = 0;
    m_write = new_size;
  }

  /*!
   * Destructor documentation which Jan Oberlaender hasn't done till now!
   */
  ~tRingBuffer()
  { }

private:
  std::vector<T> m_buffer;
  size_type m_write;
  size_type m_read;
} ICL_CORE_GCC_DEPRECATE;

}

#endif
