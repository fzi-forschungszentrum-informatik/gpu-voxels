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
 * \author  Jan Oberlaender <oberlaender@fzi.de>
 * \date    2012-11-05
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_ARRAY_2D_H_INCLUDED
#define ICL_CORE_ARRAY_2D_H_INCLUDED

#include <vector>
#include <memory>
#include <algorithm>

namespace icl_core {

/*! A simple 2D array using an STL vector container internally.
 *  \tparam T Cell data type.
 *  \tparam TAllocator Optional altervative allocator passed to the
 *          underlying std::vector.
 */
template <typename T, typename TAllocator = std::allocator<T> >
class Array2D
{
public:
  typedef std::vector<T, TAllocator> container_type;
  typedef typename container_type::value_type value_type;
  typedef typename container_type::allocator_type allocator_type;
  typedef typename container_type::size_type size_type;
  typedef typename container_type::difference_type difference_type;
  typedef typename container_type::reference reference;
  typedef typename container_type::const_reference const_reference;
  typedef typename container_type::pointer pointer;
  typedef typename container_type::const_pointer const_pointer;
  typedef typename container_type::iterator iterator;
  typedef typename container_type::const_iterator const_iterator;
  typedef typename container_type::reverse_iterator reverse_iterator;
  typedef typename container_type::const_reverse_iterator const_reverse_iterator;

  explicit Array2D(const allocator_type& alloc = allocator_type())
    : m_container(alloc),
      m_width(0),
      m_height(0)
  { }

  Array2D(size_t width, size_t height,
          const value_type& value = value_type(),
          const allocator_type& alloc = allocator_type())
    : m_container(width*height, value, alloc),
      m_width(width),
      m_height(height)
  { }

  Array2D(const Array2D& other)
    : m_container(other.m_container),
      m_width(other.m_width),
      m_height(other.m_height)
  { }

  Array2D& operator = (const Array2D& other)
  {
    m_container = other.m_container;
    m_width = other.m_width;
    m_height = other.m_height;
    return *this;
  }

  allocator_type get_allocator() const { return m_container.get_allocator(); }

  void resize(size_t width, size_t height)
  {
    m_container.resize(width*height);
    m_width = width;
    m_height = height;
  }

  void resize(size_t width, size_t height, const value_type& value)
  {
    m_container.resize(width*height, value);
    m_width = width;
    m_height = height;
  }

  void assign(size_t width, size_t height)
  {
    m_container.assign(width*height, value_type());
    m_width = width;
    m_height = height;
  }

  void assign(size_t width, size_t height, const value_type& value)
  {
    m_container.assign(width*height, value);
    m_width = width;
    m_height = height;
  }

  void reset()
  {
    m_container.assign(m_width*m_height, value_type());
  }

  void set(const value_type& value)
  {
    m_container.assign(m_width*m_height, value);
  }

  //! 1D vector-like access.
  const T& at(size_t i) const { return m_container[i]; }
  //! 1D vector-like access.
  T& at(size_t i) { return m_container[i]; }

  //! 1D vector-like access.
  const T& operator [] (size_t i) const { return m_container[i]; }
  //! 1D vector-like access.
  T& operator [] (size_t i) { return m_container[i]; }

  //! 2D access.
  const T& at(size_t column, size_t row) const { return m_container[row*m_width + column]; }
  //! 2D access.
  T& at(size_t column, size_t row) { return m_container[row*m_width + column]; }

  //! 2D access.
  const T& operator () (size_t column, size_t row) const { return m_container[row*m_width + column]; }
  //! 2D access.
  T& operator () (size_t column, size_t row) { return m_container[row*m_width + column]; }

  //! Calculate the 1D index for 2D array coordinates.
  inline size_t index1D(size_t column, size_t row) const { return row*m_width + column; }

  //! Calculate the 2D array coordinates for a given 1D index.
  inline void indexTo2D(size_t i, size_t& column, size_t& row) const
  {
    column = i % m_width;
    row = i / m_width;
  }

  //! Swap this array with another one.
  void swap(Array2D& other)
  {
    m_container.swap(other.m_container);
    std::swap(m_width, other.m_width);
    std::swap(m_height, other.m_height);
  }

  size_t width() const { return m_width; }
  size_t height() const { return m_height; }
  size_t size() const { return m_container.size(); }
  size_type max_size() const { return m_container.max_size(); }
  size_type capacity() const { return m_container.capacity(); }
  bool empty() const { return m_container.empty(); }

  iterator begin() { return m_container.begin(); }
  iterator end() { return m_container.end(); }
  const_iterator begin() const { return m_container.begin(); }
  const_iterator end() const { return m_container.end(); }

protected:
  std::vector<T> m_container;
  size_t m_width;
  size_t m_height;
};

}

#endif
