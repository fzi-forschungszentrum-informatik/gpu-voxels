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
 * \date    2007-10-03
 *
 * \brief   Contains icl_logging::LogLevel
 *
 * \b icl_logging::LogLevel
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOG_LEVEL_H_INCLUDED
#define ICL_CORE_LOGGING_LOG_LEVEL_H_INCLUDED

#include <icl_core/BaseTypes.h>

#include "icl_core_logging/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace logging {

extern ICL_CORE_LOGGING_IMPORT_EXPORT const char *log_level_descriptions[];

/*! Enumerates the available log levels.
 *
 *  The log level \a eLL_MUTE should not be used for log messages. It
 *  is intended to "mute" a log or log output stream (i.e. prevent
 *  that log messages are output).
 */
enum LogLevel
{
  eLL_TRACE,
  eLL_DEBUG,
  eLL_INFO,
  eLL_WARNING,
  eLL_ERROR,
  eLL_MUTE
};

//! The log level which is used initially.
const LogLevel cDEFAULT_LOG_LEVEL = eLL_INFO;

/*! Returns the textual description of the specified log level.
 *
 *  \param log_level The log level to be transformed into a string.
 *  \returns A textual description of the specified log level or the
 *           empty string if the log level was out of range.
 */
ICL_CORE_LOGGING_IMPORT_EXPORT const char *logLevelDescription(LogLevel log_level);

/*! Tries to convert a string into a log level.
 *
 *  \param log_level_text Input parameter containing the text to be
 *         converted.
 *  \param log_level Output parameter to which the log level is
 *         written on success.  If the conversion is not possible then
 *         this parameter is not altered.
 *  \returns \c true if the conversion was successful, \c false if the
 *           specified text did not match any defined log level.
 */
ICL_CORE_LOGGING_IMPORT_EXPORT bool stringToLogLevel(const icl_core::String& log_level_text,
                                                     LogLevel& log_level);

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Returns the textual description of the specified log level.
 *
 *  \param log_level The log level to be transformed into a string.
 *  \returns A textual description of the specified log level or the
 *           empty string if the log level was out of range.
 *  \deprecated Obsolete coding style.
 */
ICL_CORE_LOGGING_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE
const char *LogLevelDescription(LogLevel log_level) ICL_CORE_GCC_DEPRECATE_STYLE;

/*! Tries to convert a string into a log level.
 *
 *  \param log_level_text Input parameter containing the text to be
 *         converted.
 *  \param log_level Output parameter to which the log level is
 *         written on success.  If the conversion is not possible then
 *         this parameter is not altered.
 *  \returns \c true if the conversion was successful, \c false if the
 *           specified text did not match any defined log level.
 *  \deprecated Obsolete coding style.
 */
ICL_CORE_LOGGING_IMPORT_EXPORT ICL_CORE_VC_DEPRECATE_STYLE
bool StringToLogLevel(const icl_core::String& log_level_text,
                      LogLevel& log_level) ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

}
}

#endif
