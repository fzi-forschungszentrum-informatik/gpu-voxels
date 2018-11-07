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
 * \date    2014-04-04
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_EXPLICIT_H_INCLUDED
#define ICL_CORE_EXPLICIT_H_INCLUDED

#include <boost/type_traits.hpp>

namespace icl_core {

namespace internal {

//! Internal helper class for icl_core::Explicit, do not use directly.
template <bool cond> struct ExplicitEnforceMeaning
{ };
template <> struct ExplicitEnforceMeaning<true>
{
  enum { EXPLICIT_TYPE_MEANING_MUST_BE_AN_EMPTY_CLASS_TYPE = 0 };
};

}

/*! A thin wrapper around fundamental data types to give them a
 *  specific meaning and purpose.  With this, you can write functions
 *  expecting, for example, an integer with a specific meaning, and
 *  forbidding integers with different meanings.  The \a Meaning must
 *  be a "tag", i.e., an empty struct, whose name uniquely identifies
 *  the explicit class.
 *
 *  Usage example:
 *
 *  \code
 *  struct IndexTag { }; // Meaning of the value is "Index"
 *  struct FrameNumber { }; // Meaning of the value is "FrameNumber"
 *  // An integer that has the meaning "Index".
 *  typedef Explicit<int, IndexTag> Index;
 *  // An integer that has the meaning "FrameNumber".
 *  typedef Explicit<int, IndexTag> FrameNumber;
 *  ...
 *  Index i = 23, i2 = 25;
 *  FrameNumber f(42);
 *  i = 24; // This is OK
 *  i = i2; // This is also OK
 *  i = f;  // Error: incompatible types (even though both are integers).
 *  \endcode
 */
template <typename T, typename Meaning>
struct Explicit
{
  //! Default constructor does not initialize the value.
  Explicit()
  {
    (void) internal::ExplicitEnforceMeaning<boost::is_stateless<Meaning>::value>::EXPLICIT_TYPE_MEANING_MUST_BE_AN_EMPTY_CLASS_TYPE;
  }

  //! Construction from a fundamental value.
  Explicit(T value)
    : value(value)
  { }

  //! Implicit conversion back to the fundamental data type.
  inline operator T () const { return value; }

  //! The actual fundamental value.
  T value;
};

}

#endif
