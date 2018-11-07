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
 * \date    2010-07-01
 *
 * \brief   Contains a system independet PRINTF macro.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_INTERNAL_RAW_DEBUG_H_INCLUDED
#define ICL_CORE_INTERNAL_RAW_DEBUG_H_INCLUDED

#ifdef _SYSTEM_LXRT_
# include <rtai_lxrt.h>
#else
# include <stdio.h>
#endif

/*
 * This macro is only intended to be used within the implementation of icl_core and icl_core_logging.
 * DON'T USE IT IN YOUR OWN CODE! Use icl_core_logging instead!
 */
#ifdef _SYSTEM_LXRT_
# define PRINTF(arg...) do { if (icl_core::os::IsThisHRT()) { ::rt_printk(arg); } else { ::printf(arg); }} while (false)
#else
# define PRINTF ::printf
#endif

#endif
