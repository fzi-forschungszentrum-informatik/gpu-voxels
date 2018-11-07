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
 * \brief   Contains icl_logging::StdLogOutput
 *
 * \b icl_logging::StdLogOutput
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_STD_LOG_OUTPUT_H_INCLUDED
#define ICL_CORE_LOGGING_STD_LOG_OUTPUT_H_INCLUDED

#include <iostream>

#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/tLogOutputStream.h"

#define LOGGING_STDOUT ::icl_core::logging::StdLogOutput::instance()

namespace icl_core {
namespace logging {

/*! An output stream which streams to standard output.
 *
 *  This class is implemented as a singleton so that only one instance
 *  can exist in any process.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT StdLogOutput : public LogOutputStream,
                                                    protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a new STDOUT log output stream object.
   */
  static LogOutputStream *create(const icl_core::String& name, const icl_core::String& config_prefix,
                                 icl_core::logging::LogLevel log_level = cDEFAULT_LOG_LEVEL);

private:
  StdLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
               icl_core::logging::LogLevel log_level)
    : LogOutputStream(name, config_prefix, log_level)
  { }

  virtual void pushImpl(const icl_core::String& log_line);
};

}
}

#endif
