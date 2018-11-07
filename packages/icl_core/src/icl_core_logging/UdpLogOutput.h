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
 * \date    2010-06-16
 *
 * \brief   Contains icl_logging::UdpLogOutput
 *
 * \b icl_logging::UdpLogOutput
 *
 * Writes log messages to a UDP socket.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_UDP_LOG_OUTPUT_H_INCLUDED
#define ICL_CORE_LOGGING_UDP_LOG_OUTPUT_H_INCLUDED

#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogOutputStream.h"


namespace icl_core {
namespace logging {

/*! An output stream which streams to a file.
 *
 *  This class is implemented as a singleton so that only one instance
 *  can exist in any process.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT UdpLogOutput : public LogOutputStream,
                                                    protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a new file log output stream object.
   */
  static LogOutputStream *create(const icl_core::String& name, const icl_core::String& config_prefix,
                                 icl_core::logging::LogLevel log_level = cDEFAULT_LOG_LEVEL);

private:
  UdpLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
               icl_core::logging::LogLevel log_level);
  virtual ~UdpLogOutput();

  virtual void pushImpl(const LogMessage& log_message);

  icl_core::String escape(icl_core::String str) const;

  icl_core::String m_system_name;

  int m_socket;
};

}
}

#endif
