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
 * \date    2014-01-28
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_Z_BUFFER_H_INCLUDED
#define ICL_CORE_Z_BUFFER_H_INCLUDED

#include <vector>
#include <cassert>
#include <boost/optional.hpp>

namespace icl_core {

/*! A simple Z buffer implementation.  Values are stored inside the
 *  buffer only if their corresponding buffer cell is currently empty,
 *  or the values are less than the currently stored value (where the
 *  meaning of "less" can be defined by a custom operator).
 *  \tparam T The stored type.  Elements of the buffer will be of type
 *          boost::optional<T>.
 *  \tparam TCompare Comparison operator.  A stored element is updated
 *          with a new element if the comparison #m_cmp(new_elem,
 *          current_elem) returns \c true.
 */
template <typename T, typename TCompare = std::less<T> >
class ZBuffer
{
public:
  //! The type of the internal buffer.
  typedef std::vector<boost::optional<T> > Buffer;
  //! Shorthand to vector size type.
  typedef typename Buffer::size_type size_type;

  /*! Constructs a Z buffer.
   *  \param size The number of elements in the Z buffer.
   *  \param cmp Comparison operator.  Elements are replaced in the
   *         buffer if cmp(new_elem, current_elem) returns \c true.
   */
  explicit ZBuffer(const std::size_t size, const TCompare& cmp = TCompare())
    : m_buffer(size),
      m_cmp(cmp)
  { }

  /*! Updates the element at \a pos with \a value, provided the
   *  comparison operator agrees.
   *  \param pos Position to update.
   *  \param value The potential replacement.
   *  \returns \c true if the current value in the buffer was replaced
   *           with the new \a value.
   */
  bool update(const std::size_t pos, const T& value)
  {
    assert(pos < m_buffer.size());
    if (!m_buffer[pos] || m_cmp(value, *m_buffer[pos]))
    {
      m_buffer[pos] = value;
      return true;
    }
    return false;
  }

  //! Empties the Z buffer.
  void reset() { m_buffer.assign(size); }
  //! Returns the buffer size.
  const size_type size() const { return m_buffer.size(); }
  //! Read-only access to the buffer.
  const Buffer& buffer() const { return m_buffer; }
  //! Read-only access to a specific buffer element.
  boost::optional<T> at(const std::size_t pos) const { return m_buffer[pos]; }
  //! Read-only access to a specific buffer element.
  boost::optional<T> operator [] (const std::size_t pos) const { return m_buffer[pos]; }

private:
  //! The actual buffer (a vector with optionally empty elements).
  Buffer m_buffer;
  //! The comparison operator.
  TCompare m_cmp;
};

}

#endif
