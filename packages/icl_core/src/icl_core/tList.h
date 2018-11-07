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
 * \date    2008-01-29
 *
 * \brief   Contains icl_core::tList
 *
 * \b icl_core::tList
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_T_LIST_H_INCLUDED
#define ICL_CORE_T_LIST_H_INCLUDED

#include <list>

#include "icl_core/BaseTypes.h"
#include "icl_core/Deprecate.h"

namespace icl_core {

// \todo Create a wrapper class (and/or additional RT-safe implementations).
template <typename T>
class ICL_CORE_VC_DEPRECATE tList : public std::list<T>
{
} ICL_CORE_GCC_DEPRECATE;

typedef tList<uint8_t> tUnsigned8List;
typedef tList<uint16_t> tUnsigned16List;
typedef tList<uint32_t> tUnsigned32List;
typedef tList<uint64_t> tUnsigned64List;
typedef tList<int8_t> tSigned8List;
typedef tList<int16_t> tSigned16List;
typedef tList<int32_t> tSigned32List;
typedef tList<int64_t> tSigned64List;
typedef tList<float> tFloatList;
typedef tList<double> tDoubleList;

}

#endif
