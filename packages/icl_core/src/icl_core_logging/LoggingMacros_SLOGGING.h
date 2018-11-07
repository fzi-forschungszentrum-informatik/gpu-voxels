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
 * \brief   Defines SLOGGING logging macros.
 *
 * These logging macros require a log stream object as their first
 * argument.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_MACROS__SLOGGING_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MACROS__SLOGGING_H_INCLUDED

// https://www.boost.org/doc/libs/1_58_0/boost/current_function.hpp
#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) ||                      \
  (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)
#  define SLOGGING_CURRENT_FUNCTION __FUNCTION__
#elif defined(__DMC__) && (__DMC__ >= 0x810)
#  define SLOGGING_CURRENT_FUNCTION __FUNCTION__
#elif defined(__FUNCSIG__)
#  define SLOGGING_CURRENT_FUNCTION __FUNCSIG__
#elif (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) ||                                \
  (defined(__IBMCPP__) && (__IBMCPP__ >= 500))
#  define SLOGGING_CURRENT_FUNCTION __FUNCTION__
#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)
#  define SLOGGING_CURRENT_FUNCTION __FUNC__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)
#  define SLOGGING_CURRENT_FUNCTION __func__
#elif defined(__cplusplus) && (__cplusplus >= 201103)
#  define SLOGGING_CURRENT_FUNCTION __func__
#else
#  define SLOGGING_CURRENT_FUNCTION "(unknown)"
#endif

#ifdef __ANDROID__
#include <android/log.h>
#include <sstream>
#endif

#ifdef __ANDROID__
#define SLOGGING_LOG_FLCO(stream, level, filename, line, classname, objectname, arg)                                        \
  do {                                                                                                                      \
    if (stream.isActive())                                                                                                  \
    {                                                                                                                       \
      if (stream.getLogLevel() <= level)                                                                                    \
      {                                                                                                                     \
        std::stringstream str;                                                                                              \
        str << arg;                                                                                                         \
        switch (level) {                                                                                                    \
        case ::icl_core::logging::eLL_TRACE: __android_log_print(ANDROID_LOG_VERBOSE, stream.nameCStr(), str.str().c_str()); break; \
        case ::icl_core::logging::eLL_DEBUG: __android_log_print(ANDROID_LOG_DEBUG, stream.nameCStr(), str.str().c_str()); break;   \
        case ::icl_core::logging::eLL_INFO: __android_log_print(ANDROID_LOG_INFO, stream.nameCStr(), str.str().c_str()); break;     \
        case ::icl_core::logging::eLL_WARNING: __android_log_print(ANDROID_LOG_WARN, stream.nameCStr(), str.str().c_str()); break;  \
        default: __android_log_print(ANDROID_LOG_UNKNOWN, stream.nameCStr(), str.str().c_str()); break;                             \
        }                                                                                                                   \
      }                                                                                                                     \
    }                                                                                                                       \
  } while (0)
#else
#define SLOGGING_LOG_FLCO(stream, level, filename, line, classname, objectname, arg) \
  do {                                                                  \
    if (stream.isActive())                                              \
    {                                                                   \
      if (stream.getLogLevel() <= level)                                \
      {                                                                 \
        ::icl_core::logging::ThreadStream& thread_stream=stream.threadStream(level); \
        thread_stream.setLineLogLevel(level);                           \
        thread_stream.setFilename(filename);                            \
        thread_stream.setLine(line);                                    \
        thread_stream.setClassname(classname);                          \
        thread_stream.setObjectname(objectname);                        \
        thread_stream.setFunction(__func__);                            \
        thread_stream << arg;                                           \
      }                                                                 \
    }                                                                   \
  } while (0)
#endif
#define SLOGGING_LOG_CO(stream, level, classname, objectname, arg) SLOGGING_LOG_FLCO(stream, level, __FILE__, __LINE__, #classname, objectname, arg)
#define SLOGGING_LOG_C(stream, level, classname, arg) SLOGGING_LOG_FLCO(stream, level, __FILE__, __LINE__, #classname, "", arg)
#define SLOGGING_LOG(stream, level, arg) SLOGGING_LOG_FLCO(stream, level, __FILE__, __LINE__, "", "", arg)


#define SLOGGING_ERROR(stream, arg) SLOGGING_LOG(stream, ::icl_core::logging::eLL_ERROR, arg)
#define SLOGGING_WARNING(stream, arg) SLOGGING_LOG(stream, ::icl_core::logging::eLL_WARNING, arg)
#define SLOGGING_INFO(stream, arg) SLOGGING_LOG(stream, ::icl_core::logging::eLL_INFO, arg)
#ifdef _IC_DEBUG_
# define SLOGGING_DEBUG(stream, arg) SLOGGING_LOG(stream, ::icl_core::logging::eLL_DEBUG, arg)
# define SLOGGING_TRACE(stream, arg) SLOGGING_LOG(stream, ::icl_core::logging::eLL_TRACE, arg)
#else
# define SLOGGING_DEBUG(stream, arg) (void)0
# define SLOGGING_TRACE(stream, arg) (void)0
#endif


#define SLOGGING_ERROR_C(stream, classname, arg) SLOGGING_LOG_C(stream, ::icl_core::logging::eLL_ERROR, classname, arg)
#define SLOGGING_WARNING_C(stream, classname, arg) SLOGGING_LOG_C(stream, ::icl_core::logging::eLL_WARNING, classname, arg)
#define SLOGGING_INFO_C(stream, classname, arg) SLOGGING_LOG_C(stream, ::icl_core::logging::eLL_INFO,  classname, arg)
#ifdef _IC_DEBUG_
# define SLOGGING_DEBUG_C(stream, classname, arg) SLOGGING_LOG_C(stream, ::icl_core::logging::eLL_DEBUG, classname, arg)
# define SLOGGING_TRACE_C(stream, classname, arg) SLOGGING_LOG_C(stream, ::icl_core::logging::eLL_TRACE, classname, arg)
#else
# define SLOGGING_DEBUG_C(stream, classname, arg) (void)0
# define SLOGGING_TRACE_C(stream, classname, arg) (void)0
#endif


#define SLOGGING_ERROR_CO(stream, classname, objectname, arg) SLOGGING_LOG_CO(stream, ::icl_core::logging::eLL_ERROR, classname, objectname, arg)
#define SLOGGING_WARNING_CO(stream, classname, objectname, arg) SLOGGING_LOG_CO(stream, ::icl_core::logging::eLL_WARNING, classname, objectname, arg)
#define SLOGGING_INFO_CO(stream, classname, objectname, arg) SLOGGING_LOG_CO(stream, ::icl_core::logging::eLL_INFO, classname, objectname, arg)
#ifdef _IC_DEBUG_
# define SLOGGING_DEBUG_CO(stream, classname, objectname, arg) SLOGGING_LOG_CO(stream, ::icl_core::logging::eLL_DEBUG, classname, objectname, arg)
# define SLOGGING_TRACE_CO(stream, classname, objectname, arg) SLOGGING_LOG_CO(stream, ::icl_core::logging::eLL_TRACE, classname, objectname, arg)
#else
# define SLOGGING_DEBUG_CO(stream, classname, objectname, arg) (void)0
# define SLOGGING_TRACE_CO(stream, classname, objectname, arg) (void)0
#endif


#endif
