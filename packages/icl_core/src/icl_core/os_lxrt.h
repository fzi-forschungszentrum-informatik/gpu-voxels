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
 * \date    2005-11-12
 *
 * \brief   Contains global LXRT functions
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_OS_LXRT_H_INCLUDED
#define ICL_CORE_OS_LXRT_H_INCLUDED

#include <stdio.h>

#include "icl_core/ImportExport.h"
#include "icl_core/os_time.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace os {

/*!
 * Initializes the process as an LXRT process. This function should
 * be called very early in the main() function of the process.
 *
 * On systems where LXRT is not available, this function
 * does nothing.
 */
void ICL_CORE_IMPORT_EXPORT lxrtStartup();

/*!
 * Cleans up an LXRT process. This function should
 * be called very late in the main() function of the process.
 *
 * On systems where LXRT is not available, this function
 * does nothing.
 */
void ICL_CORE_IMPORT_EXPORT lxrtShutdown();

//----------------------------------------------------------------------
// Implementation
//----------------------------------------------------------------------
bool checkKernelModule(const char *name);

bool checkForLxrt(void);

bool ICL_CORE_IMPORT_EXPORT isLxrtAvailable();
bool ICL_CORE_IMPORT_EXPORT isThisLxrtTask();
bool ICL_CORE_IMPORT_EXPORT isThisHRT();

struct timeval ICL_CORE_IMPORT_EXPORT lxrtGetExecTime();

/*!
 * Returns \c true if the thread was HRT and it had to be changed to soft real time.
 */
bool ICL_CORE_IMPORT_EXPORT ensureNoHRT();

void ICL_CORE_IMPORT_EXPORT makeHRT();

int ICL_CORE_IMPORT_EXPORT makeThisAnLxrtTask();
void ICL_CORE_IMPORT_EXPORT makeThisALinuxTask();

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Initializes the process as an LXRT process.
 *  \deprecated Obsolete coding style.
 */
void ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE LxrtStartup()
  ICL_CORE_GCC_DEPRECATE_STYLE;

/*! Cleans up an LXRT process.
 *  \deprecated Obsolete coding style.
 */
void ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE LxrtShutdown()
  ICL_CORE_GCC_DEPRECATE_STYLE;

bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE CheckKernelModule(const char *name)
  ICL_CORE_GCC_DEPRECATE_STYLE;

bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE CheckForLxrt(void)
  ICL_CORE_GCC_DEPRECATE_STYLE;

bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE IsLxrtAvailable()
  ICL_CORE_GCC_DEPRECATE_STYLE;

bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE IsThisLxrtTask()
  ICL_CORE_GCC_DEPRECATE_STYLE;

bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE IsThisHRT()
  ICL_CORE_GCC_DEPRECATE_STYLE;

struct timeval ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE LxrtGetExecTime()
  ICL_CORE_GCC_DEPRECATE_STYLE;

/*! Returns \c true if the thread was HRT and it had to be changed to
 *  soft real time.
 *  \deprecated Obsolete coding style.
 */
bool ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE EnsureNoHRT()
  ICL_CORE_GCC_DEPRECATE_STYLE;

void ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE MakeHRT()
  ICL_CORE_GCC_DEPRECATE_STYLE;

int ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE MakeThisAnLxrtTask()
  ICL_CORE_GCC_DEPRECATE_STYLE;

void ICL_CORE_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE MakeThisALinuxTask()
  ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

}
}

#endif
