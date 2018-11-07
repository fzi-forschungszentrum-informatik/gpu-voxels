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
 * \date    2009-03-12
 *
 * \brief   Contains icl_core::tVector
 *
 * \b icl_core::tVector
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_T_VECTOR_H_INCLUDED
#define ICL_CORE_T_VECTOR_H_INCLUDED

#include <vector>

#include "icl_core/BaseTypes.h"
#include "icl_core/Deprecate.h"
#include "icl_core/TemplateHelper.h"

namespace icl_core {

// \todo Create a wrapper class (and/or additional RT-safe implementations).
template <typename T>
class ICL_CORE_VC_DEPRECATE tVector : public std::vector<T>
{
public:
  tVector() : std::vector<T>() { }
  tVector(const tVector& c) : std::vector<T>(c) { }
  explicit tVector(tSize num, typename ConvertToRef<T>::ToConstRef val = DefaultConstruct<T>::C())
    : std::vector<T>(num, val)
  { }
  template <typename input_iterator>
  tVector(input_iterator start, input_iterator end) : std::vector<T>(start, end) { }
} ICL_CORE_GCC_DEPRECATE;

typedef tVector<tUnsigned8> tUnsigned8Vector;
typedef tVector<tUnsigned16> tUnsigned16Vector;
typedef tVector<tUnsigned32> tUnsigned32Vector;
typedef tVector<tUnsigned64> tUnsigned64Vector;
typedef tVector<tSigned8> tSigned8Vector;
typedef tVector<tSigned16> tSigned16Vector;
typedef tVector<tSigned32> tSigned32Vector;
typedef tVector<tSigned64> tSigned64Vector;
typedef tVector<tFloat> tFloatVector;
typedef tVector<tDouble> tDoubleVector;

}

#endif
