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
 * \date    2008-04-16
 *
 * \brief   Contains constants used in icl_core::logging
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_CONSTANTS_H_INCLUDED
#define ICL_CORE_LOGGING_CONSTANTS_H_INCLUDED

namespace icl_core {
namespace logging {

/*!
 * The maximum number of characters which can be used in a single
 * log message. Surplus characters will be truncated.
 */
#ifndef cDEFAULT_LOG_SIZE
#define cDEFAULT_LOG_SIZE 2048
#endif

/*!
 * The buffer size for identifiers (log stream names, class names and function names).
 */
const size_t cMAX_IDENTIFIER_LENGTH = 256;

/*!
 * The buffer size for filenames and object descriptions.
 */
const size_t cMAX_DESCRIPTION_LENGTH = 1024;

/*!
 * The maximum number of characters which can be used for
 * extra formatting in log output streams. Surplus characters
 * will be truncated.
 */
#define cDEFAULT_OUTPUT_FORMAT_SIZE 500

/*!
 * The message queue size for log output streams when
 * using LXRT.
 */
#define cDEFAULT_FIXED_OUTPUT_STREAM_QUEUE_SIZE 1024

/*!
 * The size of the thread stream pool, which is created
 * during log stream initialization.
 */
#define cDEFAULT_LOG_THREAD_STREAM_POOL_SIZE 32

}
}

#endif
