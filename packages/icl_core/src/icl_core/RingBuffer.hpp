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
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_RING_BUFFER_HPP_INCLUDED
#define ICL_CORE_RING_BUFFER_HPP_INCLUDED

namespace icl_core {

template <typename T>
typename RingBuffer<T>::array_range RingBuffer<T>::arrayOne()
{
  if (m_write >= m_read)
  {
    return array_range(&m_buffer[m_read], m_write-m_read);
  }
  else
  {
    return array_range(&m_buffer[m_read], m_buffer.size()-m_read);
  }
}

template <typename T>
typename RingBuffer<T>::const_array_range RingBuffer<T>::arrayOne() const
{
  if (m_write >= m_read)
  {
    return const_array_range(&m_buffer[m_read], m_write-m_read);
  }
  else
  {
    return const_array_range(&m_buffer[m_read], m_buffer.size()-m_read);
  }
}

template <typename T>
typename RingBuffer<T>::array_range RingBuffer<T>::arrayTwo()
{
  if (m_write >= m_read)
  {
    return array_range(&m_buffer[m_write], 0);
  }
  else
  {
    return array_range(&m_buffer[0], m_write);
  }
}

template <typename T>
typename RingBuffer<T>::const_array_range RingBuffer<T>::arrayTwo() const
{
  if (m_write >= m_read)
  {
    return const_array_range(&m_buffer[m_write], 0);
  }
  else
  {
    return const_array_range(&m_buffer[0], m_write);
  }
}

template <typename T>
typename RingBuffer<T>::array_range RingBuffer<T>::emptyArrayOne()
{
  if (m_write >= m_read)
  {
    if (m_read == 0)
    {
      return array_range(&m_buffer[m_write], m_buffer.size()-m_write-1);
    }
    else
    {
      return array_range(&m_buffer[m_write], m_buffer.size()-m_write);
    }
  }
  else
  {
    return array_range(&m_buffer[m_write], m_read-m_write-1);
  }
}

template <typename T>
typename RingBuffer<T>::const_array_range RingBuffer<T>::emptyArrayOne() const
{
  if (m_write >= m_read)
  {
    if (m_read == 0)
    {
      return const_array_range(&m_buffer[m_write], m_buffer.size()-m_write-1);
    }
    else
    {
      return const_array_range(&m_buffer[m_write], m_buffer.size()-m_write);
    }
  }
  else
  {
    return const_array_range(&m_buffer[m_write], m_read-m_write-1);
  }
}

template <typename T>
typename RingBuffer<T>::array_range RingBuffer<T>::emptyArrayTwo()
{
  if (m_write >= m_read)
  {
    return array_range(&m_buffer[0], (m_read>0?m_read:1) - 1);
  }
  else
  {
    return array_range(&m_buffer[m_read], 0);
  }
}

template <typename T>
typename RingBuffer<T>::const_array_range RingBuffer<T>::emptyArrayTwo() const
{
  if (m_write >= m_read)
  {
    return const_array_range(&m_buffer[0], (m_read>0?m_read:1) - 1);
  }
  else
  {
    return const_array_range(&m_buffer[m_read], 0);
  }
}

template <typename T>
void RingBuffer<T>::skip(size_type count)
{
  if (count >= size())
  {
    clear();
  }
  else
  {
    m_read = (m_read+count) % m_buffer.size();
  }
}

template <typename T>
void RingBuffer<T>::fakeWrite(size_t count)
{
  if (count > reserve())
  {
    count = reserve();
  }
  m_write = (m_write + count) % m_buffer.size();
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Clears the ring buffer.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
void RingBuffer<T>::Clear()
{
  clear();
}

//! Adds an element to the ring buffer provided there is room.
/*! If \a overwrite == \c false, throws an exception if the element
 *  can not be added.  If \a overwrite == \c true, old elements are
 *  discarded to make room for new ones.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
void RingBuffer<T>::Write(const T& val, bool overwrite)
{
  write(val, overwrite);
}

//! Read an arbitrary element from the ring buffer without removing it.
/*! Throws an exception if the index is out of range.
 *  \param pos The position into the buffer. 0 is the oldest element
 *  currently present.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
const T& RingBuffer<T>::At(size_type pos) const
{
  return at(pos);
}

//! Access an arbitrary element in the ring buffer without removing it.
/*! Throws an exception if the index is out of range.
 *  \param pos The position into the buffer. 0 is the oldest element
 *  currently present.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
T& RingBuffer<T>::At(size_type pos)
{
  return at(pos);
}

/*! Removes an element from the ring buffer without returning it.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
void RingBuffer<T>::Skip()
{
  skip();
}

//! Removes an element from the ring buffer provided there is one present.
/*! Throws an exception if the buffer is empty.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
T RingBuffer<T>::Read()
{
  return read();
}

/*! Returns the current number of elements in the ring buffer.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
typename RingBuffer<T>::size_type RingBuffer<T>::Size() const
{
  return size();
}

/*! Returns the capacity of the ring buffer.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
typename RingBuffer<T>::size_type RingBuffer<T>::Capacity() const
{
  return capacity();
}

//! Changes the capacity of the ring buffer.
/*! If the new \a capacity is less than the current buffer size,
 *  only the latest \a capacity elements are kept, the rest are
 *  destroyed.
 *  \deprecated Obsolete coding style.
 */
template <typename T>
void RingBuffer<T>::SetCapacity(size_type capacity)
{
  setCapacity(capacity);
}

#endif
/////////////////////////////////////////////////

}

#endif
