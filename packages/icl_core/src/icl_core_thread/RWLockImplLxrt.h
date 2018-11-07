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
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \date    2010-02-08
 *
 * \brief   Defines icl_core::thread::RWLockImplLxrt
 *
 * \b icl_core::thread::RWLockImplLxrt
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_RWLOCK_IMPL_LXRT_H_INCLUDED
#define ICL_CORE_THREAD_RWLOCK_IMPL_LXRT_H_INCLUDED

#include "icl_core/os_lxrt.h"

#if defined(_SYSTEM_LXRT_33_)
# include "icl_core_thread/RWLockImplLxrt33.h"
#elif defined(_SYSTEM_LXRT_35_)
# include "icl_core_thread/RWLockImplLxrt35.h"
#elif defined(_SYSTEM_LXRT_38_)
# include "icl_core_thread/RWLockImplLxrt38.h"
#else
# error "Unsupported RTAI version!"
#endif

namespace icl_core {
namespace thread {

#if defined(_SYSTEM_LXRT_33_)
typedef RWLockImplLxrt33 RWLockImplLxrt;
#elif defined(_SYSTEM_LXRT_35_)
typedef RWLockImplLxrt35 RWLockImplLxrt;
#elif defined(_SYSTEM_LXRT_38_)
typedef RWLockImplLxrt38 RWLockImplLxrt;
#endif

}
}

#endif
