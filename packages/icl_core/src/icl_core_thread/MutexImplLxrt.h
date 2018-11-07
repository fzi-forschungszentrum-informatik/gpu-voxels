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
 * \date    2009-06-09
 *
 * \brief   Defines icl_core::thread::MutexImplLxrt
 *
 * \b icl_core::thread::MutexImplLxrt
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_MUTEX_IMPL_LXRT_H_INCLUDED
#define ICL_CORE_THREAD_MUTEX_IMPL_LXRT_H_INCLUDED

#include "icl_core/os_lxrt.h"

#if defined(_SYSTEM_LXRT_33_)
# include "icl_core_thread/MutexImplLxrt33.h"
#elif defined(_SYSTEM_LXRT_35_)
# include "icl_core_thread/MutexImplLxrt35.h"
#elif defined(_SYSTEM_LXRT_38_)
# include "icl_core_thread/MutexImplLxrt38.h"
#else
# error "Unsupported RTAI version!"
#endif

namespace icl_core {
namespace thread {

#ifdef _SYSTEM_LXRT_33_
typedef MutexImplLxrt33 MutexImplLxrt;
#elif defined(_SYSTEM_LXRT_35_)
typedef MutexImplLxrt35 MutexImplLxrt;
#elif defined(_SYSTEM_LXRT_38_)
typedef MutexImplLxrt38 MutexImplLxrt;
#endif

}
}

#endif
