// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-04-14
 *
 * \brief   Contains logging definitions for the icl_core_thread library.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_LOGGING_H_INCLUDED
#define ICL_CORE_THREAD_LOGGING_H_INCLUDED

#include "icl_core_logging/Logging.h"

namespace icl_core {
namespace thread {

DECLARE_LOG_STREAM(IclCoreThread)

using icl_core::logging::endl;

}
}

#endif
