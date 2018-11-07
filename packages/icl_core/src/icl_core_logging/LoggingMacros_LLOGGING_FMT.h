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
#ifndef ICL_CORE_LOGGING_LOGGING_MACROS__LLOGGING__FMT_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MACROS__LLOGGING__FMT_H_INCLUDED

#include "icl_core_logging/LoggingMacros_LOGGING_FMT.h"


#ifdef ICL_CORE_LOCAL_LOGGING
# define LLOGGING_FMT_ERROR(streamname, ...) LOGGING_LOG(streamname, ::icl_core::logging::eLL_ERROR, __VA_ARGS__)
# define LLOGGING_FMT_WARNING(streamname, ...) LOGGING_LOG(streamname, ::icl_core::logging::eLL_WARNING, __VA_ARGS__)
# define LLOGGING_FMT_INFO(streamname, ...) LOGGING_LOG(streamname, ::icl_core::logging::eLL_INFO, __VA_ARGS__)
# ifdef _IC_DEBUG_
#  define LLOGGING_FMT_DEBUG(streamname, ...) LOGGING_LOG(streamname, ::icl_core::logging::eLL_DEBUG, __VA_ARGS__)
#  define LLOGGING_FMT_TRACE(streamname, ...) LOGGING_LOG(streamname, ::icl_core::logging::eLL_TRACE, __VA_ARGS__)
# else
#  define LLOGGING_FMT_DEBUG(streamname, ...) (void)0
#  define LLOGGING_FMT_TRACE(streamname, ...) (void)0
# endif
#else
# define LLOGGING_FMT_ERROR(streamname, ...) (void)0
# define LLOGGING_FMT_WARNING(streamname, ...) (void)0
# define LLOGGING_FMT_INFO(streamname, ...) (void)0
# define LLOGGING_FMT_DEBUG(streamname, ...) (void)0
# define LLOGGING_FMT_TRACE(streamname, ...) (void)0
#endif


#ifdef ICL_CORE_LOCAL_LOGGING
# define LLOGGING_FMT_ERROR_C(streamname, classname, ...) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_ERROR, classname, __VA_ARGS__)
# define LLOGGING_FMT_WARNING_C(streamname, classname, ...) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_WARNING, classname, __VA_ARGS__)
# define LLOGGING_FMT_INFO_C(streamname, classname, ...) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_INFO,  classname, __VA_ARGS__)
# ifdef _IC_DEBUG_
#  define LLOGGING_FMT_DEBUG_C(streamname, classname, ...) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_DEBUG, classname, __VA_ARGS__)
#  define LLOGGING_FMT_TRACE_C(streamname, classname, ...) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_TRACE, classname, __VA_ARGS__)
# else
#  define LLOGGING_FMT_DEBUG_C(streamname, classname, ...) (void)0
#  define LLOGGING_FMT_TRACE_C(streamname, classname, ...) (void)0
# endif
#else
# define LLOGGING_FMT_ERROR_C(streamname, classname, ...) (void)0
# define LLOGGING_FMT_WARNING_C(streamname, classname, ...) (void)0
# define LLOGGING_FMT_INFO_C(streamname, classname, ...) (void)0
# define LLOGGING_FMT_DEBUG_C(streamname, classname, ...) (void)0
# define LLOGGING_FMT_TRACE_C(streamname, classname, ...) (void)0
#endif


#ifdef ICL_CORE_LOCAL_LOGGING
# define LLOGGING_FMT_ERROR_CO(streamname, classname, objectname, ...) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_ERROR, classname, objectname, function, __VA_ARGS__)
# define LLOGGING_FMT_WARNING_CO(streamname, classname, objectname, ...) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_WARNING, classname, objectname, function, __VA_ARGS__)
# define LLOGGING_FMT_INFO_CO(streamname, classname, objectname, ...) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_INFO, classname, objectname, function, __VA_ARGS__)
# ifdef _IC_DEBUG_
#  define LLOGGING_FMT_DEBUG_CO(streamname, classname, objectname, ...) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_DEBUG, classname, objectname, function, __VA_ARGS__)
#  define LLOGGING_FMT_TRACE_CO(streamname, classname, objectname, ...) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_TRACE, classname, objectname, function, __VA_ARGS__)
# else
#  define LLOGGING_FMT_DEBUG_CO(streamname, classname, objectname, ...) (void)0
#  define LLOGGING_FMT_TRACE_CO(streamname, classname, objectname, ...) (void)0
# endif
#else
# define LLOGGING_FMT_ERROR_CO(streamname, classname, objectname, ...) (void)0
# define LLOGGING_FMT_WARNING_CO(streamname, classname, objectname, ...) (void)0
# define LLOGGING_FMT_INFO_CO(streamname, classname, objectname, ...) (void)0
# define LLOGGING_FMT_DEBUG_CO(streamname, classname, objectname, ...) (void)0
# define LLOGGING_FMT_TRACE_CO(streamname, classname, objectname, ...) (void)0
#endif

#endif
