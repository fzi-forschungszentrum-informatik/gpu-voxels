// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 */
//----------------------------------------------------------------------
#include <iostream>

#include "icl_core_logging/Logging.h"
#include "icl_core_logging/StdLogOutput.h"

namespace icl_core {
namespace logging {

REGISTER_LOG_OUTPUT_STREAM(Stdout, &StdLogOutput::create)

LogOutputStream *StdLogOutput::create(const icl_core::String& name, const icl_core::String& config_prefix,
                                      icl_core::logging::LogLevel log_level)
{
  return new StdLogOutput(name, config_prefix, log_level);
}

void StdLogOutput::pushImpl(const icl_core::String& log_line)
{
  std::cout << log_line;
}

}
}
