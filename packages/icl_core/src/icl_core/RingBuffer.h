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
#ifndef ICL_CORE_RING_BUFFER_H_INCLUDED
#define ICL_CORE_RING_BUFFER_H_INCLUDED

#include <stdexcept>
#include <iterator>
#include <vector>

#include "icl_core/BaseTypes.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

/*! \brief A simple ring buffer implementation based on std::vector.
 *
 *  \note This ring buffer is only suitable for scalar or POD data
 *        types!  This is because elements are not deconstructed upon
 *        removal, except when the RingBuffer itself is destroyed.
 */
template <typename T>
class RingBuffer
{
public:
  typedef T value_type;
  typedef typename std::vector<T>::size_type size_type;
  typedef typename std::vector<T>::difference_type difference_type;
  //! Array range as in boost::circular_buffer::array_range.
  typedef std::pair<T*, size_type> array_range;
  //! Array range as in boost::circular_buffer::const_array_range.
  typedef std::pair<const T*, size_type> const_array_range;
  static const size_type cDEFAULT_CAPACITY = 32;

  //! Const iterator for RingBuffers.
  class const_iterator : public std::iterator<std::random_access_iterator_tag, T>
  {
    friend class RingBuffer;

  public:
    const_iterator(const const_iterator& other)
      : m_current(other.m_current), m_cbegin(other.m_cbegin), m_cend(other.m_cend), m_begin(other.m_begin)
    { }

    const_iterator& operator = (const const_iterator& other)
    {
      m_current = other.m_current; m_cbegin = other.m_cbegin; m_cend = other.m_cend; m_begin = other.m_begin;
      return *this;
    }

    const_iterator& operator ++ ()
    {
      ++m_current;
      if (m_current == m_cend) { m_current = m_cbegin; }
      return *this;
    }
    const_iterator operator ++ (int)
    {
      const_iterator answer = *this;
      operator ++ ();
      return answer;
    }

    const_iterator& operator -- ()
    {
      if (m_current == m_cbegin)
      {
        m_current = m_cend;
      }
      --m_current;
      return *this;
    }
    const_iterator operator -- (int)
    {
      const_iterator answer = *this;
      operator -- ();
      return answer;
    }

    const_iterator& operator += (difference_type offset)
    {
      m_current += offset;
      if (m_current >= m_cend)
      {
        m_current -= m_cend - m_cbegin;
      }
      if (m_cbegin > m_current)
      {
        m_current += m_cend - m_cbegin;
      }
      return *this;
    }
    const_iterator& operator -= (difference_type offset)
    {
      return operator += (-offset);
    }
    const_iterator operator + (difference_type offset) const
    {
      const_iterator answer = *this;
      answer += offset;
      return answer;
    }
    const_iterator operator - (difference_type offset) const
    {
      const_iterator answer = *this;
      answer -= offset;
      return answer;
    }

    difference_type operator - (const const_iterator& other) const
    {
      if ((m_current >= m_begin && other.m_current >= other.m_begin)
          || (m_current < m_begin && other.m_current < other.m_begin))
      {
        return m_current - other.m_current;
      }
      else if (m_current >= m_begin)
      {
        return m_current - other.m_current + m_cbegin - m_cend;
      }
      else
      {
        return m_current - other.m_current + m_cend - m_cbegin;
      }
    }

    bool operator == (const const_iterator& other) const { return m_current == other.m_current; }
    bool operator != (const const_iterator& other) const { return m_current != other.m_current; }

    const T& operator * () const { return *m_current; }
    const T *operator -> () const { return m_current; }

  protected:
    const_iterator(const T *current, const T *cbegin, const T *cend, const T *begin)
      : m_current(current), m_cbegin(cbegin), m_cend(cend), m_begin(begin)
    { }

    const T *m_current;
    //! Beginning of the container.
    const T *m_cbegin;
    //! End of the container.
    const T *m_cend;
    //! Actual first value in the container.
    const T *m_begin;
  };

  /*! Iterator for RingBuffers.  Extends const_iterator by providing
   *  non-const access to the underlying pointed-to element.  This is
   *  accomplished via const_cast<>() in order to avoid code
   *  duplication.
   */
  class iterator : public const_iterator
  {
    friend class RingBuffer;

  public:
    iterator(const iterator& other)
      : const_iterator(other)
    { }

    iterator& operator = (const iterator& other)
    {
      const_iterator::m_current = other.m_current;
      const_iterator::m_cbegin = other.m_cbegin;
      const_iterator::m_cend = other.m_cend;
      const_iterator::m_begin = other.m_begin;
      return *this;
    }

    T& operator * () const { return *const_cast<T *>(const_iterator::m_current); }
    T *operator -> () const { return const_cast<T *>(const_iterator::m_current); }

  protected:
    iterator(const T *current, const T *cbegin, const T *cend, const T *begin)
      : const_iterator(current, cbegin, cend, begin)
    { }
  };

  //! Default constructor.
  RingBuffer(size_type capacity = cDEFAULT_CAPACITY)
    : m_buffer(capacity+1), m_write(0), m_read(0)
  { }

  //! Copy constructor.
  RingBuffer(const RingBuffer<T>& other)
    : m_buffer(other.m_buffer), m_write(other.m_write), m_read(other.m_read)
  { }

  //! Assignment operator.
  RingBuffer& operator = (const RingBuffer<T>& other)
  {
    m_buffer = other.m_buffer;
    m_write = other.m_write;
    m_read = other.m_read;
    return *this;
  }

  //! Clears the ring buffer.
  void clear() { m_write = 0; m_read = 0; }

  //! Returns \c true if the buffer is empty.
  bool empty() const { return m_write == m_read; }

  //! Returns \c true if the buffer is full.
  bool full() const { return ((m_write+1) % m_buffer.size()) == m_read; }

  //! Adds an element to the ring buffer provided there is room.
  /*! If \a overwrite == \c false, throws an exception if the element
   *  can not be added.  If \a overwrite == \c true, old elements are
   *  discarded to make room for new ones.
   */
  void write(const T& val, bool overwrite=false)
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
        skip();
      }
      else
      {
        throw std::out_of_range("RingBuffer::write: capacity exceeded");
      }
    }
    m_buffer[m_write] = val;
    m_write = new_write_pos;
  }

  /*! Increases ring buffer size by \a count, at most up to its
   *  capacity.  The added elements are not overwritten.  Use this if
   *  you have filled buffer elements (from emptyArrayOne() and
   *  emptyArrayTwo()) manually, to adjust the ring buffer's write
   *  pointer.
   */
  void fakeWrite(size_t count);

  //! Read an arbitrary element from the ring buffer without removing it.
  /*! Throws an exception if the index is out of range.
   *  \param pos The position into the buffer. 0 is the oldest element
   *  currently present.
   */
  const T& at(size_type pos) const
  {
    if (pos < size())
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
      throw std::out_of_range("RingBuffer::at: out of range");
    }
  }

  //! Access an arbitrary element in the ring buffer without removing it.
  /*! Throws an exception if the index is out of range.
   *  \param pos The position into the buffer. 0 is the oldest element
   *  currently present.
   */
  T& at(size_type pos)
  {
    if (pos < size())
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
      throw std::out_of_range("RingBuffer::at: out of range");
    }
  }

  const T& front() const { return at(0); }
  T& front() { return at(0); }

  //! Removes an element from the ring buffer without returning it.
  void skip()
  {
    if (m_write == m_read)
    {
      throw std::out_of_range("RingBuffer::skip: buffer empty");
    }
    m_read++;
    if (m_read >= m_buffer.size())
    {
      m_read = 0;
    }
  }

  /*! Removes a number of elements from the ring buffer without
   *  returning them.  If the ring buffer is empty, nothing happens.
   */
  void skip(size_type count);

  //! Removes an element from the ring buffer provided there is one present.
  /*! Throws an exception if the buffer is empty.
   */
  T read()
  {
    if (m_write == m_read)
    {
      throw std::out_of_range("RingBuffer::read: buffer empty");
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
  inline size_type size() const
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
  size_type capacity() const { return m_buffer.size()-1; }

  //! Returns the remaining reserve (free space) of the ring buffer.
  size_type reserve() const
  {
    return capacity() - size();
  }

  //! Changes the capacity of the ring buffer.
  /*! If the new \a capacity is less than the current buffer size,
   *  only the latest \a capacity elements are kept, the rest are
   *  destroyed.
   */
  void setCapacity(size_type capacity)
  {
    size_type old_size = size();
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
  ~RingBuffer()
  { }

  const_iterator begin() const
  { return const_iterator(&m_buffer[m_read], &m_buffer[0], &m_buffer[0]+m_buffer.size(), &m_buffer[m_read]); }
  iterator begin()
  { return iterator(&m_buffer[m_read], &m_buffer[0], &m_buffer[0]+m_buffer.size(), &m_buffer[m_read]); }

  const_iterator end() const
  { return const_iterator(&m_buffer[m_write], &m_buffer[0], &m_buffer[0]+m_buffer.size(), &m_buffer[m_read]); }
  iterator end()
  { return iterator(&m_buffer[m_write], &m_buffer[0], &m_buffer[0]+m_buffer.size(), &m_buffer[m_read]); }

  /*! Returns a pointer to the first contiguous filled block of data
   *  in the RingBuffer, and the block length.  Since the RingBuffer
   *  is a contiguous block of memory, but its filled area may wrap
   *  around from the end of that block to the beginning, there may be
   *  two separate blocks to consider.  arrayOne() provides access to
   *  the first block, while arrayTwo() provides access to the second
   *  block.  Similarly, there may be two empty blocks: the first at
   *  the end of the internal buffer, after the RingBuffer's last
   *  element, and the second at the beginning of the internal buffer,
   *  before the RingBuffer's first element.  emptyArrayOne() and
   *  emptyArrayTwo() provide access to those blocks.
   *
   *  These access methods are provided for old C APIs and external
   *  routines which need raw buffer pointers to write into.
   */
  array_range arrayOne();
  /*! Returns a pointer to the first contiguous filled block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  const_array_range arrayOne() const;

  /*! Returns a pointer to the second contiguous filled block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  array_range arrayTwo();
  /*! Returns a pointer to the second contiguous filled block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  const_array_range arrayTwo() const;

  /*! Returns a pointer to the first contiguous empty block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  array_range emptyArrayOne();
  /*! Returns a pointer to the first contiguous empty block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  const_array_range emptyArrayOne() const;
  /*! Returns a pointer to the second contiguous empty block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  array_range emptyArrayTwo();
  /*! Returns a pointer to the second contiguous empty block of data
   *  in the RingBuffer, and the block length.
   *  \see arrayOne()
   */
  const_array_range emptyArrayTwo() const;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Clears the ring buffer.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Clear() ICL_CORE_GCC_DEPRECATE_STYLE;

  //! Adds an element to the ring buffer provided there is room.
  /*! If \a overwrite == \c false, throws an exception if the element
   *  can not be added.  If \a overwrite == \c true, old elements are
   *  discarded to make room for new ones.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE
  void Write(const T& val, bool overwrite=false) ICL_CORE_GCC_DEPRECATE_STYLE;

  //! Read an arbitrary element from the ring buffer without removing it.
  /*! Throws an exception if the index is out of range.
   *  \param pos The position into the buffer. 0 is the oldest element
   *  currently present.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE const T& At(size_type pos) const ICL_CORE_GCC_DEPRECATE_STYLE;

  //! Access an arbitrary element in the ring buffer without removing it.
  /*! Throws an exception if the index is out of range.
   *  \param pos The position into the buffer. 0 is the oldest element
   *  currently present.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE T& At(size_type pos) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Removes an element from the ring buffer without returning it.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Skip() ICL_CORE_GCC_DEPRECATE_STYLE;

  //! Removes an element from the ring buffer provided there is one present.
  /*! Throws an exception if the buffer is empty.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE T Read() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the current number of elements in the ring buffer.
   *  \deprecated Obsolete coding style.
   */
  inline ICL_CORE_VC_DEPRECATE_STYLE size_type Size() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the capacity of the ring buffer.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE size_type Capacity() const ICL_CORE_GCC_DEPRECATE_STYLE;

  //! Changes the capacity of the ring buffer.
  /*! If the new \a capacity is less than the current buffer size,
   *  only the latest \a capacity elements are kept, the rest are
   *  destroyed.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetCapacity(size_type capacity) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  std::vector<T> m_buffer;
  size_type m_write;
  size_type m_read;
};

}

#include "icl_core/RingBuffer.hpp"

#endif
