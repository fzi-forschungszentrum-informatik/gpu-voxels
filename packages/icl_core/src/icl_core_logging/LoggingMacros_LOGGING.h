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
 * \date    2010-04-27
 *
 * \brief   Defines logging macros.
 *
 * These logging macros require the name of a log stream as their first
 * argument.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_MACROS__LOGGING_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MACROS__LOGGING_H_INCLUDED

#include "icl_core_logging/LoggingMacros_SLOGGING.h"

#define LOGGING_LOG_FLCO(streamname, level, filename, line, classname, objectname, arg) \
  do {                                                                  \
    ::icl_core::logging::LogStream& stream = streamname::instance();    \
    SLOGGING_LOG_FLCO(stream, level, filename, line, classname, objectname, arg); \
  } while (0)
#define LOGGING_LOG_CO(streamname, level, classname, objectname, arg) LOGGING_LOG_FLCO(streamname, level, __FILE__, __LINE__, #classname, objectname, arg)
#define LOGGING_LOG_C(streamname, level, classname, arg) LOGGING_LOG_FLCO(streamname, level, __FILE__, __LINE__, #classname, "", arg)
#define LOGGING_LOG(streamname, level, arg) LOGGING_LOG_FLCO(streamname, level, __FILE__, __LINE__, "", "", arg)


#define LOGGING_ERROR(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_ERROR, arg)
#define LOGGING_WARNING(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_WARNING, arg)
#define LOGGING_INFO(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_INFO, arg)
#ifdef _IC_DEBUG_
# define LOGGING_DEBUG(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_DEBUG, arg)
# define LOGGING_TRACE(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_TRACE, arg)
#else
# define LOGGING_DEBUG(streamname, arg) (void)0
# define LOGGING_TRACE(streamname, arg) (void)0
#endif


#define LOGGING_ERROR_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_ERROR, classname, arg)
#define LOGGING_WARNING_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_WARNING, classname, arg)
#define LOGGING_INFO_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_INFO,  classname, arg)
#ifdef _IC_DEBUG_
# define LOGGING_DEBUG_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_DEBUG, classname, arg)
# define LOGGING_TRACE_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_TRACE, classname, arg)
#else
# define LOGGING_DEBUG_C(streamname, classname, arg) (void)0
# define LOGGING_TRACE_C(streamname, classname, arg) (void)0
#endif


#define LOGGING_ERROR_CO(stream, classname, objectname, arg) LOGGING_LOG_CO(stream, ::icl_core::logging::eLL_ERROR, classname, objectname, arg)
#define LOGGING_WARNING_CO(stream, classname, objectname, arg) LOGGING_LOG_CO(stream, ::icl_core::logging::eLL_WARNING, classname, objectname, arg)
#define LOGGING_INFO_CO(stream, classname, objectname, arg) LOGGING_LOG_CO(stream, ::icl_core::logging::eLL_INFO, classname, objectname, arg)
#ifdef _IC_DEBUG_
# define LOGGING_DEBUG_CO(stream, classname, objectname, arg) LOGGING_LOG_CO(stream, ::icl_core::logging::eLL_DEBUG, classname, objectname, arg)
# define LOGGING_TRACE_CO(stream, classname, objectname, arg) LOGGING_LOG_CO(stream, ::icl_core::logging::eLL_TRACE, classname, objectname, arg)
#else
# define LOGGING_DEBUG_CO(stream, classname, objectname, arg) (void)0
# define LOGGING_TRACE_CO(stream, classname, objectname, arg) (void)0
#endif


#endif
