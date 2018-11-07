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
 * \date    2011-04-07
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_VECTOR_H_INCLUDED
#define ICL_CORE_VECTOR_H_INCLUDED

#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/TemplateHelper.h"

namespace icl_core {

// \todo Create a wrapper class (and/or additional RT-safe implementations).
template <typename T>
class Vector : public std::vector<T>
{
public:
  typedef typename std::vector<T>::size_type size_type;

  Vector() : std::vector<T>() {}
  Vector(const Vector& c) : std::vector<T>(c) { }
  Vector(const std::vector<T>& c) : std::vector<T>(c) { }
  explicit Vector(size_type num, typename ConvertToRef<T>::ToConstRef val = DefaultConstruct<T>::C())
    : std::vector<T>(num, val)
  { }
  template <typename input_iterator>
  Vector(input_iterator start, input_iterator end) : std::vector<T>(start, end) { }
};

typedef Vector<uint8_t> Unsigned8Vector;
typedef Vector<uint16_t> Unsigned16Vector;
typedef Vector<uint32_t> Unsigned32Vector;
typedef Vector<uint64_t> Unsigned64Vector;
typedef Vector<int8_t> Signed8Vector;
typedef Vector<int16_t> Signed16Vector;
typedef Vector<int32_t> Signed32Vector;
typedef Vector<int64_t> Signed64Vector;
typedef Vector<float> FloatVector;
typedef Vector<double> DoubleVector;

}

#endif
