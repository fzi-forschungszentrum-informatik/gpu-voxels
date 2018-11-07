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
 * \date    2010-02-22
 *
 * \brief   Contains helper functions to handle bitfields.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_BITFIELD_HELPER_H_INCLUDED
#define ICL_CORE_BITFIELD_HELPER_H_INCLUDED

#include "icl_core/TemplateHelper.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

/*!
 * Sets the specified \a bits in the \a bitfield.
 */
template <typename T>
void setBits(int bits, typename ConvertToRef<T>::ToRef bitfield)
{
  bitfield = T(int(bitfield) | bits);
}

/*!
 * Clears the specified \a bits in the \a bitfield.
 */
template <typename T>
void clearBits(int bits, typename ConvertToRef<T>::ToRef bitfield)
{
  bitfield = T(int(bitfield) & ~bits);
}

/*!
 * Checks if the specified \a bit is set in the \a bitfield.
 */
template <typename T>
bool isBitSet(int bit, typename ConvertToRef<T>::ToConstRef bitfield)
{
  return (bitfield & T(bit)) != 0;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*!
 * Sets the specified \a bits in the \a bitfield.
 * \deprecated Obsolete coding style.
 */
template <typename T>
void SetBits(int bits, typename ConvertToRef<T>::ToRef bitfield) ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE void SetBits(int bits, typename ConvertToRef<T>::ToRef bitfield)
{
  setBits(bits, bitfield);bitfield = T(int(bitfield) | bits);
}

/*!
 * Clears the specified \a bits in the \a bitfield.
 * \deprecated Obsolete coding style.
 */
template <typename T>
void ClearBits(int bits, typename ConvertToRef<T>::ToRef bitfield) ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE void ClearBits(int bits, typename ConvertToRef<T>::ToRef bitfield)
{
  clearBits(bits, bitfield);
}

/*!
 * Checks if the specified \a bit is set in the \a bitfield.
 * \deprecated Obsolete coding style.
 */
template <typename T>
bool IsBitSet(int bit, typename ConvertToRef<T>::ToConstRef bitfield) ICL_CORE_GCC_DEPRECATE_STYLE;
template <typename T>
ICL_CORE_VC_DEPRECATE_STYLE bool IsBitSet(int bit, typename ConvertToRef<T>::ToConstRef bitfield)
{
  return isBitSet(bit, bitfield);
}

#endif
/////////////////////////////////////////////////

}

#endif
