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
 * \date    2010-04-21
 *
 * \brief   Collects all exported header files for use with precompiled headers.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_ICL_CORE_THREAD_H_INCLUDED
#define ICL_CORE_THREAD_ICL_CORE_THREAD_H_INCLUDED

#ifndef _IC_BUILDER_ICL_CORE_
#  define _IC_BUILDER_ICL_CORE_
#endif

#include <icl_core/icl_core.h>
#include <icl_core_logging/icl_core_logging.h>

#include "icl_core_thread/Mutex.h"
#include "icl_core_thread/PeriodicThread.h"
#include "icl_core_thread/RWLock.h"
#include "icl_core_thread/ScopedMutexLock.h"
#include "icl_core_thread/ScopedRWLock.h"
#include "icl_core_thread/Sem.h"
#include "icl_core_thread/Thread.h"
#include "icl_core_thread/tMutex.h"
#include "icl_core_thread/tPeriodicThread.h"
#include "icl_core_thread/tRWLock.h"
#include "icl_core_thread/tScopedMutexLock.h"
#include "icl_core_thread/tScopedRWLock.h"
#include "icl_core_thread/tSemaphore.h"
#include "icl_core_thread/tThread.h"

#endif
