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
#ifndef ICL_CORE_LOGGING_LOGGING_MACROS__SLOGGING__FMT_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MACROS__SLOGGING__FMT_H_INCLUDED

#define SLOGGING_FMT_LOG_FLCO(stream, level, filename, line, classname, objectname, ...) \
  do {                                                                  \
    if (stream.isActive())                                              \
    {                                                                   \
      if (stream.getLogLevel() <= level)                                \
      {                                                                 \
        ::icl_core::logging::ThreadStream& thread_stream = stream.threadStream(level); \
        thread_stream.setLineLogLevel(level);                           \
        thread_stream.setFilename(filename);                            \
        thread_stream.setLine(line);                                    \
        thread_stream.setClassname(classname);                          \
        thread_stream.setObjectname(objectname);                        \
        thread_stream.setFunction(__FUNCTION__);                        \
        thread_stream.printf(__VA_ARGS__);                              \
      }                                                                 \
    }                                                                   \
  } while (0)
#define SLOGGING_FMT_LOG_CO(stream, level, classname, objectname, ...) SLOGGING_FMT_LOG_FLCO(stream, level, __FILE__, __LINE__, #classname, objectname, __VA_ARGS__)
#define SLOGGING_FMT_LOG_C(stream, level, classname, ...) SLOGGING_FMT_LOG_FLCO(stream, level, __FILE__, __LINE__, #classname, "", __VA_ARGS__)
#define SLOGGING_FMT_LOG(stream, level, ...) SLOGGING_FMT_LOG_FLCO(stream, level, __FILE__, __LINE__, "", "", __VA_ARGS__)


#define SLOGGING_FMT_ERROR(stream, ...) SLOGGING_FMT_LOG(stream, ::icl_core::logging::eLL_ERROR, __VA_ARGS__)
#define SLOGGING_FMT_WARNING(stream, ...) SLOGGING_FMT_LOG(stream, ::icl_core::logging::eLL_WARNING, __VA_ARGS__)
#define SLOGGING_FMT_INFO(stream, ...) SLOGGING_FMT_LOG(stream, ::icl_core::logging::eLL_INFO, __VA_ARGS__)
#ifdef _IC_DEBUG_
# define SLOGGING_FMT_DEBUG(stream, ...) SLOGGING_FMT_LOG(stream, ::icl_core::logging::eLL_DEBUG, __VA_ARGS__)
# define SLOGGING_FMT_TRACE(stream, ...) SLOGGING_FMT_LOG(stream, ::icl_core::logging::eLL_TRACE, __VA_ARGS__)
#else
# define SLOGGING_FMT_DEBUG(stream, ...) (void)0
# define SLOGGING_FMT_TRACE(stream, ...) (void)0
#endif


#define SLOGGING_FMT_ERROR_C(stream, classname, ...) SLOGGING_FMT_LOG_C(stream, ::icl_core::logging::eLL_ERROR, classname, __VA_ARGS__)
#define SLOGGING_FMT_WARNING_C(stream, classname, ...) SLOGGING_FMT_LOG_C(stream, ::icl_core::logging::eLL_WARNING, classname, __VA_ARGS__)
#define SLOGGING_FMT_INFO_C(stream, classname, ...) SLOGGING_FMT_LOG_C(stream, ::icl_core::logging::eLL_INFO,  classname, __VA_ARGS__)
#ifdef _IC_DEBUG_
# define SLOGGING_FMT_DEBUG_C(stream, classname, ...) SLOGGING_FMT_LOG_C(stream, ::icl_core::logging::eLL_DEBUG, classname, __VA_ARGS__)
# define SLOGGING_FMT_TRACE_C(stream, classname, ...) SLOGGING_FMT_LOG_C(stream, ::icl_core::logging::eLL_TRACE, classname, __VA_ARGS__)
#else
# define SLOGGING_FMT_DEBUG_C(stream, classname, ...) (void)0
# define SLOGGING_FMT_TRACE_C(stream, classname, ...) (void)0
#endif


#define SLOGGING_FMT_ERROR_CO(stream, classname, objectname, ...) SLOGGING_FMT_LOG_CO(stream, ::icl_core::logging::eLL_ERROR, classname, objectname, __VA_ARGS__)
#define SLOGGING_FMT_WARNING_CO(stream, classname, objectname, ...) SLOGGING_FMT_LOG_CO(stream, ::icl_core::logging::eLL_WARNING, classname, objectname, __VA_ARGS__)
#define SLOGGING_FMT_INFO_CO(stream, classname, objectname, ...) SLOGGING_FMT_LOG_CO(stream, ::icl_core::logging::eLL_INFO, classname, objectname, __VA_ARGS__)
#ifdef _IC_DEBUG_
# define SLOGGING_FMT_DEBUG_CO(stream, classname, objectname, ...) SLOGGING_FMT_LOG_CO(stream, ::icl_core::logging::eLL_DEBUG, classname, objectname, __VA_ARGS__)
# define SLOGGING_FMT_TRACE_CO(stream, classname, objectname, ...) SLOGGING_FMT_LOG_CO(stream, ::icl_core::logging::eLL_TRACE, classname, objectname, __VA_ARGS__)
#else
# define SLOGGING_FMT_DEBUG_CO(stream, classname, objectname, ...) (void)0
# define SLOGGING_FMT_TRACE_CO(stream, classname, objectname, ...) (void)0
#endif

#endif
