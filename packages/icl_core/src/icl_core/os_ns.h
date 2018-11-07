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
 * \date    2008-01-28
 *
 * \brief   Definition of the implementation namespace
 *          for global functions.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_NS_H_INCLUDED
#define ICL_CORE_OS_NS_H_INCLUDED

#if defined _SYSTEM_POSIX_
# define ICL_CORE_OS_IMPL_NS ::icl_core::os::hidden_posix
#elif defined _SYSTEM_WIN32_
# define ICL_CORE_OS_IMPL_NS ::icl_core::os::hidden_win32
#else
# error "No os namespace implementation defined for this platform."
#endif

#endif
