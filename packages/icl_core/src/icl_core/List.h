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
 * \date    2011.04-07
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LIST_H_INCLUDED
#define ICL_CORE_LIST_H_INCLUDED

#include <list>

#include "icl_core/BaseTypes.h"
#include "icl_core/TemplateHelper.h"

namespace icl_core {

// \todo Create a wrapper class (and/or additional RT-safe implementations).
template <typename T>
class List : public std::list<T>
{
public:
  typedef typename std::list<T>::size_type size_type;

  List() : std::list<T>() { }
  List(const List& c) : std::list<T>(c) { }
  List(const std::list<T>& c) : std::list<T>(c) { }
  explicit List(size_type num, typename ConvertToRef<T>::ToConstRef val = DefaultConstruct<T>::C())
    : std::list<T>(num, val)
  { }
  template <typename TInputIterator>
  List(TInputIterator start, TInputIterator end) : std::list<T>(start, end) { }
};

typedef List<uint8_t> Unsigned8List;
typedef List<uint16_t> Unsigned16List;
typedef List<uint32_t> Unsigned32List;
typedef List<uint64_t> Unsigned64List;
typedef List<int8_t> Signed8List;
typedef List<int16_t> Signed16List;
typedef List<int32_t> Signed32List;
typedef List<int64_t> Signed64List;
typedef List<float> FloatList;
typedef List<double> DoubleList;

}

#endif
