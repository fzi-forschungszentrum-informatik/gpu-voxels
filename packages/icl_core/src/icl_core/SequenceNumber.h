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
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-07-22
 *
 * \brief   Contains SequenceNumber
 *
 * \b SequenceNumber
 *
 * Implements a sequence number with a maximum value.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SEQUENCE_NUMBER_H_INCLUDED
#define ICL_CORE_SEQUENCE_NUMBER_H_INCLUDED

#include "icl_core/BaseTypes.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

/*! Implements a sequence number which can wrap around with the
 *  internal type \a TBase, the minimum value \a min_value, the
 *  maximum value \a max_value and the initial value \a initial_value.
 */
template <typename TBase, TBase max_value, TBase min_value = 0, TBase initial_value = min_value>
class SequenceNumber
{
public:
  /*!
   * Constructs a new sequence number.
   * \param value The sequence number value.
   */
  explicit SequenceNumber(TBase value = initial_value)
    : m_value(value)
  {
  }

  /*!
   * Constructs a new sequence number as a copy of \a other.
   */
  SequenceNumber(const SequenceNumber& other)
    : m_value(other.m_value)
  {
  }

  /*!
   * Assigns a sequence number.
   */
  SequenceNumber& operator = (const SequenceNumber& other)
  {
    m_value = other.m_value;
    return *this;
  }

  /*!
   * Assigns the raw data type.
   */
  SequenceNumber& operator = (TBase value)
  {
    m_value = value;
    return *this;
  }

  /*!
   * Checks if \a this sequence number is lower than the \a other.
   */
  bool operator < (const SequenceNumber& other) const
  {
    if (m_value < other.m_value)
    {
      return (other.m_value - m_value) < max_value / 2;
    }
    else
    {
      return (m_value - other.m_value) > max_value / 2;
    }
  }

  /*!
   * Checks if \a this sequence number is lower than or equal to the \a other.
   */
  bool operator <= (const SequenceNumber& other) const
  {
    return (*this) < other || (*this) == other;
  }

  /*!
   * Checks if \a this sequence number is greater than the \a other.
   */
  bool operator > (const SequenceNumber& other) const
  {
    return (*this) != other && !((*this) < other);
  }

  /*!
   * Checks if \a this sequence number is greater than or equal to the \a other.
   */
  bool operator >= (const SequenceNumber& other) const
  {
    return !((*this) < other);
  }

  /*!
   * Compares two sequence numbers for equality.
   */
  bool operator == (const SequenceNumber& other) const
  {
    return m_value == other.m_value;
  }

  /*!
   * Compares two sequence numbers for inequality.
   */
  bool operator != (const SequenceNumber& other) const
  {
    return !((*this) == other);
  }

  /*!
   * Prefix increment operator.
   */
  SequenceNumber& operator ++ ()
  {
    ++m_value;
    if (m_value >= max_value)
    {
      m_value = min_value;
    }
    return *this;
  }

  /*!
   * Postfix increment operator.
   */
  SequenceNumber operator ++ (int)
  {
    SequenceNumber result = *this;
    ++(*this);
    return result;
  }

  /*!
   * Implicit conversion to TBase.
   */
  operator TBase () const
  {
    return m_value;
  }

  /*!
   * Get the maximum sequence number value.
   */
  static TBase maxValue()
  {
    return max_value;
  }

  /*!
   * Get the minimum sequence number value.
   */
  static TBase minValue()
  {
    return min_value;
  }

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*!
   * Get the maximum sequence number value.
   * \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE TBase MaxValue() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*!
   * Get the minimum sequence number value.
   * \deprecated Obsolete coding style.
   */
  static ICL_CORE_VC_DEPRECATE_STYLE TBase MinValue() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

private:
  TBase m_value;
};

}

#endif
