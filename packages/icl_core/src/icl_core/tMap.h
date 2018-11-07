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
 * \author  Thomas Schamm <schamm@fzi.de>
 * \date    2008-05-29
 *
 * \brief   Contains icl_core::tMap
 *
 * \b icl_core::tMap
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_T_MAP_H_INCLUDED
#define ICL_CORE_T_MAP_H_INCLUDED

#include <map>

#include "icl_core/Deprecate.h"

namespace icl_core
{

// \todo Create a wrapper class (and/or additional RT-safe implementations).
template <typename KEY, typename VALUE>
class ICL_CORE_VC_DEPRECATE tMap : public std::map<KEY, VALUE>
{
} ICL_CORE_GCC_DEPRECATE;

}

#endif
