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
 * \date    2006-05-10
 *
 * \brief   Contains icl_core::logging::StdErrorLogOutput
 *
 * \b icl_core::logging::StdErrorLogOutput
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_STD_ERROR_LOG_OUTPUT_H_INCLUDED
#define ICL_CORE_LOGGING_STD_ERROR_LOG_OUTPUT_H_INCLUDED

#include <iostream>

#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogOutputStream.h"

#define LOGGING_STDERR ::icl_core::logging::StdErrorLogOutput::instance()

namespace icl_core {
namespace logging {

/*! An output stream which streams to std err.
 *
 *  This class is implemented as a singleton so that only one instance
 *  can exist in any process.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT StdErrorLogOutput : public LogOutputStream,
                                                         protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a new STDERR log output stream.
   */
  static LogOutputStream *create(const icl_core::String& name, const icl_core::String& config_prefix,
                                 icl_core::logging::LogLevel log_level = cDEFAULT_LOG_LEVEL);

private:
  StdErrorLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
                    icl_core::logging::LogLevel log_level)
    : LogOutputStream(name, config_prefix, log_level)
  { }

  virtual void pushImpl(const icl_core::String& log_line);
};

}
}

#endif
