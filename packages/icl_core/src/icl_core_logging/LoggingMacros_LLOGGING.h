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
 * \brief   Defines LLOGGING logging macros.
 *
 * These macros are only activated if ICL_CORE_LOCAL_LOGGING has been
 * define prior to including "icl_core_logging/Logging.h". Therefore
 * logging with these macros can be compiled in and out on a per file
 * basis.
 */
//----------------------------------------------------------------------
// No header guards as this may be included multiple times!
#include "icl_core_logging/LoggingMacros_LOGGING.h"

// Undef the macros first
#undef LLOGGING_ERROR
#undef LLOGGING_WARNING
#undef LLOGGING_INFO
#undef LLOGGING_DEBUG
#undef LLOGGING_TRACE

#undef LLOGGING_ERROR_C
#undef LLOGGING_WARNING_C
#undef LLOGGING_INFO_C
#undef LLOGGING_DEBUG_C
#undef LLOGGING_TRACE_C

#undef LLOGGING_ERROR_CO
#undef LLOGGING_WARNING_CO
#undef LLOGGING_INFO_CO
#undef LLOGGING_DEBUG_CO
#undef LLOGGING_TRACE_CO

// And then redefine them given the current local logging setting.

#ifdef ICL_CORE_LOCAL_LOGGING
# define LLOGGING_ERROR(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_ERROR, arg)
# define LLOGGING_WARNING(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_WARNING, arg)
# define LLOGGING_INFO(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_INFO, arg)
# ifdef _IC_DEBUG_
#  define LLOGGING_DEBUG(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_DEBUG, arg)
#  define LLOGGING_TRACE(streamname, arg) LOGGING_LOG(streamname, ::icl_core::logging::eLL_TRACE, arg)
# else
#  define LLOGGING_DEBUG(streamname, arg) (void)0
#  define LLOGGING_TRACE(streamname, arg) (void)0
# endif
#else
# define LLOGGING_ERROR(streamname, arg) (void)0
# define LLOGGING_WARNING(streamname, arg) (void)0
# define LLOGGING_INFO(streamname, arg) (void)0
# define LLOGGING_DEBUG(streamname, arg) (void)0
# define LLOGGING_TRACE(streamname, arg) (void)0
#endif


#ifdef ICL_CORE_LOCAL_LOGGING
# define LLOGGING_ERROR_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_ERROR, classname, arg)
# define LLOGGING_WARNING_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_WARNING, classname, arg)
# define LLOGGING_INFO_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_INFO,  classname, arg)
# ifdef _IC_DEBUG_
#  define LLOGGING_DEBUG_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_DEBUG, classname, arg)
#  define LLOGGING_TRACE_C(streamname, classname, arg) LOGGING_LOG_C(streamname, ::icl_core::logging::eLL_TRACE, classname, arg)
# else
#  define LLOGGING_DEBUG_C(streamname, classname, arg) (void)0
#  define LLOGGING_TRACE_C(streamname, classname, arg) (void)0
# endif
#else
# define LLOGGING_ERROR_C(streamname, classname, arg) (void)0
# define LLOGGING_WARNING_C(streamname, classname, arg) (void)0
# define LLOGGING_INFO_C(streamname, classname, arg) (void)0
# define LLOGGING_DEBUG_C(streamname, classname, arg) (void)0
# define LLOGGING_TRACE_C(streamname, classname, arg) (void)0
#endif


#ifdef ICL_CORE_LOCAL_LOGGING
# define LLOGGING_ERROR_CO(streamname, classname, objectname, arg) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_ERROR, classname, objectname, function, arg)
# define LLOGGING_WARNING_CO(streamname, classname, objectname, arg) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_WARNING, classname, objectname, function, arg)
# define LLOGGING_INFO_CO(streamname, classname, objectname, arg) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_INFO, classname, objectname, function, arg)
# ifdef _IC_DEBUG_
#  define LLOGGING_DEBUG_CO(streamname, classname, objectname, arg) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_DEBUG, classname, objectname, function, arg)
#  define LLOGGING_TRACE_CO(streamname, classname, objectname, arg) LOGGING_LOG_COF(streamname, ::icl_core::logging::eLL_TRACE, classname, objectname, function, arg)
# else
#  define LLOGGING_DEBUG_CO(streamname, classname, objectname, arg) (void)0
#  define LLOGGING_TRACE_CO(streamname, classname, objectname, arg) (void)0
# endif
#else
# define LLOGGING_ERROR_CO(streamname, classname, objectname, arg) (void)0
# define LLOGGING_WARNING_CO(streamname, classname, objectname, arg) (void)0
# define LLOGGING_INFO_CO(streamname, classname, objectname, arg) (void)0
# define LLOGGING_DEBUG_CO(streamname, classname, objectname, arg) (void)0
# define LLOGGING_TRACE_CO(streamname, classname, objectname, arg) (void)0
#endif
