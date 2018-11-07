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
 * \brief   Contains mocos::tString
 *
 * \b mocos::tString
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_T_STRING_H_INCLUDED
#define ICL_CORE_T_STRING_H_INCLUDED

#include <string>

#include "icl_core/Deprecate.h"

namespace icl_core {

typedef ICL_CORE_VC_DEPRECATE std::string tString ICL_CORE_GCC_DEPRECATE;

}

#endif
