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
 * \date    2014-04-08
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_FINALIZABLE_H_INCLUDED
#define ICL_CORE_FINALIZABLE_H_INCLUDED

#include <iostream>
#include <stdexcept>

namespace icl_core {

/*! A simple object wrapper for objects which may be modified until
 *  they are "finalized", after which they are constant.
 *  Finalizable<T> behaves mostly like T, but after calling
 *  finalize(), assignments are no longer allowed and result in a
 *  std::logic_error.
 *
 *  This is useful in cases where variables cannot be made const, but
 *  should have a const behavior once they have received their correct
 *  value.
 */
template <typename T>
class Finalizable
{
public:
  Finalizable(const T& value)
    : m_value(value),
      m_final(false)
  { }

  /*! Copy construction includes the #m_final state, so a finalized
   *  object remains finalized.
   */
  Finalizable(const Finalizable& other)
    : m_value(other.m_value),
      m_final(other.m_final)
  { }

  //! Assignment is only allowed as long as this is not final.
  Finalizable& operator = (const Finalizable& other)
  {
    if (!m_final)
    {
      m_value = other.m_value;
      return *this;
    }
    else
    {
      throw std::logic_error("object is final");
    }
  }

  //! Assigning a value is only allowed as long as this is not final.
  Finalizable& operator = (const T& value)
  {
    if (!m_final)
    {
      m_value = value;
      return *this;
    }
    else
    {
      throw std::logic_error("object is final");
    }
  }

  //! Implicit conversion.
  operator T () const { return m_value; }

  //! Returns \c true if the object is final.
  bool isFinal() const { return m_final; }

  //! Finalizes the object, i.e. makes it unmodifiable.
  void finalize() { m_final = true; }

  //! Output stream operator.
  std::ostream& operator << (std::ostream& os)
  {
    return os << m_value;
  }

private:
  /*! The value managed by this object.  Can be modified as long as
   *  #m_final is \c false, after that it appears constant.
   */
  T m_value;
  //! Indicates whether #m_value is final, i.e. no longer modifiable.
  bool m_final;
};

}

#endif
