// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-07-02
 *
 */
//----------------------------------------------------------------------
#include "icl_core_logging/SQLiteLogOutput.h"

#include <iostream>

#include "icl_core_config/Config.h"
#include "icl_core_logging/Logging.h"

namespace icl_core {
namespace logging {

REGISTER_LOG_OUTPUT_STREAM(SQLite, &SQLiteLogOutput::create)

LogOutputStream *SQLiteLogOutput::create(const icl_core::String& name, const icl_core::String& config_prefix,
                                         icl_core::logging::LogLevel log_level)
{
  return new SQLiteLogOutput(name, config_prefix, log_level);
}

SQLiteLogOutput::SQLiteLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
                                 icl_core::logging::LogLevel log_level)
  : LogOutputStream(name, config_prefix, log_level),
    m_db(NULL)
{
  icl_core::String db_filename = "";
  if (!icl_core::config::get<icl_core::String>(config_prefix + "/FileName", db_filename))
  {
    std::cerr << "SQLite log output: No filename specified for SQLite log output stream "
              << config_prefix << std::endl;
  }

  bool rotate = false;
  icl_core::config::get<bool>(config_prefix + "/Rotate", rotate);

  m_db = new SQLiteLogDb(db_filename, rotate);
}

SQLiteLogOutput::~SQLiteLogOutput()
{
  delete m_db;
  m_db = NULL;
}

void SQLiteLogOutput::onStart()
{
  m_db->openDatabase();
}

void SQLiteLogOutput::pushImpl(const LogMessage& log_message)
{
  m_db->writeLogLine("", log_message.timestamp.formatIso8601().c_str(), log_message.log_stream,
                     logLevelDescription(log_message.log_level), log_message.filename,
                     log_message.line, log_message.class_name, log_message.object_name,
                     log_message.function_name, log_message.message_text);
}

void SQLiteLogOutput::onShutdown()
{
  m_db->closeDatabase();
}

}
}
