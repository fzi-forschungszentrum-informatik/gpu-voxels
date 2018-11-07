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
 * \date    2008-12-19
 *
 * \brief   Defines icl_core::logging::SemaphoreImplLxrt
 *
 * \b icl_core::logging::SemaphoreImplLxrt
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SEMAPHORE_IMPL_LXRT_H_INCLUDED
#define ICL_CORE_LOGGING_SEMAPHORE_IMPL_LXRT_H_INCLUDED

#include "icl_core/os_lxrt.h"

#if defined(_SYSTEM_LXRT_33_)
# include "SemaphoreImplLxrt33.h"
#elif defined(_SYSTEM_LXRT_35_)
# include "SemaphoreImplLxrt35.h"
#elif defined(_SYSTEM_LXRT_38_)
# include "SemaphoreImplLxrt38.h"
#else
# error "Unsupported RTAI version!"
#endif

namespace icl_core {
namespace logging {

#if defined(_SYSTEM_LXRT_33_)
typedef SemaphoreImplLxrt33 SemaphoreImplLxrt;
#elif defined(_SYSTEM_LXRT_35_)
typedef SemaphoreImplLxrt35 SemaphoreImplLxrt;
#elif defined(_SYSTEM_LXRT_38_)
typedef SemaphoreImplLxrt38 SemaphoreImplLxrt;
#endif

}
}

#endif
