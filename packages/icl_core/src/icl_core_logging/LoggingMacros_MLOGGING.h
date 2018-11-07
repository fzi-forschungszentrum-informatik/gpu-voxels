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
 * \brief   Defines MLOGGING logging macros.
 *
 * These logging macros require that a Debug() function is callable from
 * the context, from where the macros are called. Log messages are only
 * output if the Debug() function returns \a true.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_MACROS__MLOGGING_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MACROS__MLOGGING_H_INCLUDED

#include "icl_core_logging/LoggingMacros_SLOGGING.h"

#define MLOGGING_LOG_FLCO(streamname, level, filename, line, classname, objectname, arg) \
  do {                                                                  \
    if (Debug())                                                        \
    {                                                                   \
      ::icl_core::logging::LogStream& stream = streamname::instance();  \
      SLOGGING_LOG_FLCO(stream, level, filename, line, classname, objectname, arg); \
    }                                                                   \
  } while (0)
#define MLOGGING_LOG_COF(streamname, level, classname, objectname, function, arg) MLOGGING_LOG_FLCO(streamname, level, __FILE__, __LINE__, #classname, objectname, arg)
#define MLOGGING_LOG_C(streamname, level, classname, arg) MLOGGING_LOG_FLCO(streamname, level, __FILE__, __LINE__, #classname, "", arg)
#define MLOGGING_LOG(streamname, level, arg) MLOGGING_LOG_FLCO(streamname, level, __FILE__, __LINE__, "", "", arg)


#define MLOGGING_ERROR(streamname, arg) MLOGGING_LOG(streamname, ::icl_core::logging::eLL_ERROR, arg)
#define MLOGGING_WARNING(streamname, arg) MLOGGING_LOG(streamname, ::icl_core::logging::eLL_WARNING, arg)
#define MLOGGING_INFO(streamname, arg) MLOGGING_LOG(streamname, ::icl_core::logging::eLL_INFO, arg)
#ifdef _IC_DEBUG_
# define MLOGGING_DEBUG(streamname, arg) MLOGGING_LOG(streamname, ::icl_core::logging::eLL_DEBUG, arg)
# define MLOGGING_TRACE(streamname, arg) MLOGGING_LOG(streamname, ::icl_core::logging::eLL_TRACE, arg)
#else
# define MLOGGING_DEBUG(streamname, arg) (void)0
# define MLOGGING_TRACE(streamname, arg) (void)0
#endif


#define MLOGGING_ERROR_C(streamname, classname, arg) MLOGGING_LOG_C(streamname, ::icl_core::logging::eLL_ERROR, classname, arg)
#define MLOGGING_WARNING_C(streamname, classname, arg) MLOGGING_LOG_C(streamname, ::icl_core::logging::eLL_WARNING, classname, arg)
#define MLOGGING_INFO_C(streamname, classname, arg) MLOGGING_LOG_C(streamname, ::icl_core::logging::eLL_INFO,  classname, arg)
#ifdef _IC_DEBUG_
# define MLOGGING_DEBUG_C(streamname, classname, arg) MLOGGING_LOG_C(streamname, ::icl_core::logging::eLL_DEBUG, classname, arg)
# define MLOGGING_TRACE_C(streamname, classname, arg) MLOGGING_LOG_C(streamname, ::icl_core::logging::eLL_TRACE, classname, arg)
#else
# define MLOGGING_DEBUG_C(streamname, classname, arg) (void)0
# define MLOGGING_TRACE_C(streamname, classname, arg) (void)0
#endif


#define MLOGGING_ERROR_CO(streamname, classname, objectname, arg) MLOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_ERROR, classname, objectname, function, arg)
#define MLOGGING_WARNING_CO(streamname, classname, objectname, arg) MLOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_WARNING, classname, objectname, function, arg)
#define MLOGGING_INFO_CO(streamname, classname, objectname, arg) MLOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_INFO, classname, objectname, function, arg)
#ifdef _IC_DEBUG_
# define MLOGGING_DEBUG_CO(streamname, classname, objectname, arg) MLOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_DEBUG, classname, objectname, function, arg)
# define MLOGGING_TRACE_CO(streamname, classname, objectname, arg) MLOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_TRACE, classname, objectname, function, arg)
#else
# define MLOGGING_DEBUG_CO(streamname, classname, objectname, arg) (void)0
# define MLOGGING_TRACE_CO(streamname, classname, objectname, arg) (void)0
#endif


#endif
