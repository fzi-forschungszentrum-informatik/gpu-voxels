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
 * \date    2008-10-26
 *
 * \brief   Collects all exported header files for use with precompiled headers.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_ICL_CORE_H_INCLUDED
#define ICL_CORE_ICL_CORE_H_INCLUDED

#ifndef _IC_BUILDER_ICL_CORE_
#  define _IC_BUILDER_ICL_CORE_
#endif

/*! \namespace icl_core
 *
 *  Independent Component Library containing core functionality.
 */

#include "icl_core/BaseTypes.h"
#include "icl_core/BitfieldHelper.h"
#include "icl_core/KeyValueDirectory.h"
#include "icl_core/KeyValueDirectory.hpp"
#include "icl_core/List.h"
#include "icl_core/Map.h"
#include "icl_core/Noncopyable.h"
#include "icl_core/os.h"
#include "icl_core/os_fs.h"
#include "icl_core/os_lxrt.h"
#include "icl_core/os_mem.h"
#include "icl_core/os_ns.h"
#include "icl_core/os_string.h"
#include "icl_core/os_thread.h"
#include "icl_core/os_time.h"
#include "icl_core/Queue.h"
#include "icl_core/RingBuffer.h"
#include "icl_core/SearchableStack.h"
#include "icl_core/StringHelper.h"
#include "icl_core/tSequenceNumber.h"
#include "icl_core/Singleton.h"
#include "icl_core/TemplateHelper.h"
#include "icl_core/TimeSpan.h"
#include "icl_core/TimeStamp.h"
#include "icl_core/Vector.h"

#endif
