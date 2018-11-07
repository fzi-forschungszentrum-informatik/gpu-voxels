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
 * \date    2011-08-17
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_MACROS__LOGGING__FMT_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MACROS__LOGGING__FMT_H_INCLUDED

#include "icl_core_logging/LoggingMacros_SLOGGING_FMT.h"

#define LOGGING_FMT_LOG_FLCO(streamname, level, filename, line, classname, objectname, ...) \
  do {                                                                  \
    ::icl_core::logging::LogStream& stream = streamname::instance();    \
    SLOGGING_FMT_LOG_FLCO(stream, level, filename, line, classname, objectname, __VA_ARGS__); \
  } while (0)
#define LOGGING_FMT_LOG_CO(streamname, level, classname, objectname, ...) LOGGING_FMT_LOG_FLCO(streamname, level, __FILE__, __LINE__, #classname, objectname, __VA_ARGS__)
#define LOGGING_FMT_LOG_C(streamname, level, classname, ...) LOGGING_FMT_LOG_FLCO(streamname, level, __FILE__, __LINE__, #classname, "", __VA_ARGS__)
#define LOGGING_FMT_LOG(streamname, level, ...) LOGGING_FMT_LOG_FLCO(streamname, level, __FILE__, __LINE__, "", "", __VA_ARGS__)


#define LOGGING_FMT_ERROR(streamname, ...) LOGGING_FMT_LOG(streamname, icl_core::logging::eLL_ERROR, __VA_ARGS__)
#define LOGGING_FMT_WARNING(streamname, ...) LOGGING_FMT_LOG(streamname, icl_core::logging::eLL_WARNING, __VA_ARGS__)
#define LOGGING_FMT_INFO(streamname, ...) LOGGING_FMT_LOG(streamname, icl_core::logging::eLL_INFO, __VA_ARGS__)
#ifdef _IC_DEBUG_
# define LOGGING_FMT_DEBUG(streamname, ...) LOGGING_FMT_LOG(streamname, icl_core::logging::eLL_DEBUG, __VA_ARGS__)
# define LOGGING_FMT_TRACE(streamname, ...) LOGGING_FMT_LOG(streamname, icl_core::logging::eLL_TRACE, __VA_ARGS__)
#else
# define LOGGING_FMT_DEBUG(streamname, ...) (void)0
# define LOGGING_FMT_TRACE(streamname, ...) (void)0
#endif


#define LOGGING_FMT_ERROR_C(streamname, classname, ...) LOGGING_FMT_LOG_C(streamname, ::icl_core::logging::eLL_ERROR, classname, __VA_ARGS__)
#define LOGGING_FMT_WARNING_C(streamname, classname, ...) LOGGING_FMT_LOG_C(streamname, ::icl_core::logging::eLL_WARNING, classname, __VA_ARGS__)
#define LOGGING_FMT_INFO_C(streamname, classname, ...) LOGGING_FMT_LOG_C(streamname, ::icl_core::logging::eLL_INFO,  classname, __VA_ARGS__)
#ifdef _IC_DEBUG_
# define LOGGING_FMT_DEBUG_C(streamname, classname, ...) LOGGING_FMT_LOG_C(streamname, ::icl_core::logging::eLL_DEBUG, classname, __VA_ARGS__)
# define LOGGING_FMT_TRACE_C(streamname, classname, ...) LOGGING_FMT_LOG_C(streamname, ::icl_core::logging::eLL_TRACE, classname, __VA_ARGS__)
#else
# define LOGGING_FMT_DEBUG_C(streamname, classname, ...) (void)0
# define LOGGING_FMT_TRACE_C(streamname, classname, ...) (void)0
#endif


#define LOGGING_FMT_ERROR_CO(streamname, classname, objectname, ...) LOGGING_FMT_LOG_CO(streamname, ::icl_core::logging::eLL_ERROR, classname, objectname, __VA_ARGS__)
#define LOGGING_FMT_WARNING_CO(streamname, classname, objectname, ...) LOGGING_FMT_LOG_CO(streamname, ::icl_core::logging::eLL_WARNING, classname, objectname, __VA_ARGS__)
#define LOGGING_FMT_INFO_CO(streamname, classname, objectname, ...) LOGGING_FMT_LOG_CO(streamname, ::icl_core::logging::eLL_INFO, classname, objectname, __VA_ARGS__)
#ifdef _IC_DEBUG_
# define LOGGING_FMT_DEBUG_CO(streamname, classname, objectname, ...) LOGGING_FMT_LOG_CO(streamname, ::icl_core::logging::eLL_DEBUG, classname, objectname, __VA_ARGS__)
# define LOGGING_FMT_TRACE_CO(streamname, classname, objectname, ...) LOGGING_FMT_LOG_CO(streamname, ::icl_core::logging::eLL_TRACE, classname, objectname, __VA_ARGS__)
#else
# define LOGGING_FMT_DEBUG_CO(streamname, classname, objectname, ...) (void)0
# define LOGGING_FMT_TRACE_CO(streamname, classname, objectname, ...) (void)0
#endif

#endif
