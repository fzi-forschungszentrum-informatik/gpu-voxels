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
 * \date    2009-07-02
 *
 * \brief   Contains icl_logging::SQLiteLogOutput
 *
 * \b icl_logging::SQLiteLogOutput
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_SQLITE_LOG_OUTPUT_H_INCLUDED
#define ICL_CORE_LOGGING_SQLITE_LOG_OUTPUT_H_INCLUDED

#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogOutputStream.h"
#include "icl_core_logging/SQLiteLogDb.h"

namespace icl_core {
namespace logging {

/*! An output stream which writes log messages to a SQLite database.
 *
 *  This class is implemented as a singleton so that only one instance
 *  can exist in any process.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT SQLiteLogOutput : public LogOutputStream,
                                                       protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a new SQLite log output stream object.
   */
  static LogOutputStream *create(const icl_core::String& name,
                                 const icl_core::String& config_prefix,
                                 icl_core::logging::LogLevel log_level = cDEFAULT_LOG_LEVEL);

private:
  SQLiteLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
                  icl_core::logging::LogLevel log_level);
  virtual ~SQLiteLogOutput();

  virtual void onStart();
  virtual void pushImpl(const LogMessage& log_message);
  virtual void onShutdown();

  SQLiteLogDb *m_db;
};

}
}

#endif
