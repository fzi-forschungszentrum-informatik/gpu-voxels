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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 * \brief   Contains Interface base classes and base types
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_BASE_TYPES_H_INCLUDED
#define ICL_CORE_BASE_TYPES_H_INCLUDED

#include <string>

#if defined(_MSC_VER) && _MSC_VER < 1600
# include "icl_core/msvc_stdint.h"
#else
# include <stddef.h>
# include <stdint.h>
#endif

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/tString.h"
#endif

namespace icl_core {
typedef std::string String;
}

#ifdef _IC_BUILDER_ENABLE_BASE_TYPES_

#ifdef _IC_BUILDER_DEPRECATED_BASE_TYPES_
#include "icl_core/Deprecate.h"
# define BASE_TYPES_VC_DEPRECATE(arg) ICL_CORE_VC_DEPRECATE_COMMENT(arg)
# define BASE_TYPES_GCC_DEPRECATE(arg) ICL_CORE_GCC_DEPRECATE_COMMENT(arg)
#else
# define BASE_TYPES_VC_DEPRECATE(arg)
# define BASE_TYPES_GCC_DEPRECATE(arg)
#endif

#undef max

typedef bool
  BASE_TYPES_VC_DEPRECATE("use bool instead")
  tBool
  BASE_TYPES_GCC_DEPRECATE("use bool instead");

typedef float
  BASE_TYPES_VC_DEPRECATE("use float instead")
  tFloat
  BASE_TYPES_GCC_DEPRECATE("use float instead");
typedef double
  BASE_TYPES_VC_DEPRECATE("use double instead")
  tDouble
  BASE_TYPES_GCC_DEPRECATE("use double instead");

typedef
  BASE_TYPES_VC_DEPRECATE("use uint8_t instead")
  uint8_t tUnsigned8
  BASE_TYPES_GCC_DEPRECATE("use uint8_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use int8_t instead")
  int8_t tSigned8
  BASE_TYPES_GCC_DEPRECATE("use int8_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use uint16_t instead")
  uint16_t tUnsigned16
  BASE_TYPES_GCC_DEPRECATE("use uint16_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use int16_t instead")
  int16_t tSigned16
  BASE_TYPES_GCC_DEPRECATE("use int16_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use uint32_t instead")
  uint32_t tUnsigned32
  BASE_TYPES_GCC_DEPRECATE("use uint32_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use int32_t instead")
  int32_t tSigned32
  BASE_TYPES_GCC_DEPRECATE("use int32_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use uint64_t instead")
  uint64_t tUnsigned64
  BASE_TYPES_GCC_DEPRECATE("use uint64_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use int64_t instead")
  int64_t tSigned64
  BASE_TYPES_GCC_DEPRECATE("use int64_t instead");

typedef
  BASE_TYPES_VC_DEPRECATE("use size_t instead")
  size_t tSize
  BASE_TYPES_GCC_DEPRECATE("use size_t instead");
typedef
  BASE_TYPES_VC_DEPRECATE("use ptrdiff_t instead")
  ptrdiff_t tSSize
  BASE_TYPES_GCC_DEPRECATE("use ptrdiff_t instead");

#endif

typedef uint16_t tChangedCounter;

// This is for Qt translation stuff!
#ifndef QT_TR_NOOP
# define QT_TR_NOOP(x) (x)
#endif
#ifndef QT_TRANSLATE_NOOP
# define QT_TRANSLATE_NOOP(scope, x) (x)
#endif

#endif
