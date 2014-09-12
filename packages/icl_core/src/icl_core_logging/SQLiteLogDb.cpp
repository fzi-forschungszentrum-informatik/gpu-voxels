// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2009-07-03
 *
 */
//----------------------------------------------------------------------
#include "SQLiteLogDb.h"

#include <iostream>

namespace icl_core {
namespace logging {

icl_core::String SQLiteLogDb::m_create_sql =
  "CREATE TABLE log_entries (seq INTEGER PRIMARY KEY, app_id TEXT, "
  "timestamp TIMESTAMP, log_stream TEXT, log_level TEXT, filename TEXT, "
  "line INTEGER, class_name TEXT, object_name TEXT, function_name TEXT, message TEXT)";
icl_core::String SQLiteLogDb::m_insert_sql =
  "INSERT INTO log_entries (app_id, timestamp, log_stream, log_level, "
  "filename, line, class_name, object_name, function_name, message) "
  "VALUES (:app_id, :timestamp, :log_stream, :log_level, :filename, "
  ":line, :class_name, :object_name, :function_name, :message)";

SQLiteLogDb::SQLiteLogDb(const icl_core::String& db_filename, bool rotate)
  : m_db_filename(db_filename),
    m_db(NULL),
    m_insert_stmt(NULL),
    m_rotate(rotate),
    m_last_rotation(icl_core::TimeStamp::now().days())
{
}

SQLiteLogDb::~SQLiteLogDb()
{
  closeDatabase();
}

void SQLiteLogDb::openDatabase()
{
  char *error = NULL;

  if (m_db_filename != "")
  {
    int res = SQLITE_OK;
    sqlite3_stmt *query_sql = NULL;

    // Try to open the database.
    res = sqlite3_open(m_db_filename.c_str(), &m_db);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not open SQLite database "
                << m_db_filename << ": " << sqlite3_errmsg(m_db) << std::endl;
      goto fail_return;
    }

    res = sqlite3_prepare_v2(m_db,
                             "SELECT sql FROM sqlite_master WHERE type='table' AND name='log_entries'",
                             -1, &query_sql, NULL);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not check if the log table exists in "
                << m_db_filename << ": " << sqlite3_errmsg(m_db) << std::endl;
      goto fail_return;
    }

    res = sqlite3_step(query_sql);
    if (res == SQLITE_DONE)
    {
      if (sqlite3_exec(m_db, m_create_sql.c_str(), NULL, NULL, &error) != SQLITE_OK)
      {
        std::cerr << "SQLite log output: Could not create the log table: " << error << std::endl;
        sqlite3_free(error);
        sqlite3_finalize(query_sql);
        goto fail_return;
      }
    }

    sqlite3_finalize(query_sql);

    // If we reach this point then the database is ready for action,
    // so we prepare the insert statement.
    res = sqlite3_prepare_v2(m_db, m_insert_sql.c_str(), -1, &m_insert_stmt, NULL);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not prepare the insert statement: "
                << sqlite3_errmsg(m_db) << std::endl;
      goto fail_return;
    }

    // Finally, we set some PRAGMAs to speed up operation.
    error = NULL;
    res = sqlite3_exec(m_db, "PRAGMA synchronous=OFF", NULL, NULL, &error);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not set PRAGMA synchronous=OFF: " << error << std::endl;
    }

    error = NULL;
    res = sqlite3_exec(m_db, "PRAGMA temp_store=MEMORY", NULL, NULL, &error);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not set PRAGMA temp_store=MEMORY: " << error << std::endl;
    }
  }

  return;
fail_return:
  closeDatabase();
}

void SQLiteLogDb::closeDatabase()
{
  if (m_insert_stmt != NULL)
  {
    sqlite3_finalize(m_insert_stmt);
    m_insert_stmt = NULL;
  }

  if (m_db != NULL)
  {
    sqlite3_close(m_db);
    m_db = NULL;
  }
}

void SQLiteLogDb::writeLogLine(const char *app_id, const char *timestamp, const char *log_stream,
                               const char *log_level, const char *filename,
                               size_t line, const char *class_name, const char *object_name,
                               const char *function_name, const char *message_text)
{
  if (m_rotate)
  {
    int64_t current_day = icl_core::TimeStamp::now().days();
    if (m_last_rotation != current_day)
    {
      m_last_rotation = current_day;

      closeDatabase();

      char time_str[11];
      icl_core::TimeStamp::now().strfTime(time_str, 11, "%Y-%m-%d");
      rename(m_db_filename.c_str(), (m_db_filename + "." + time_str).c_str());

      openDatabase();
    }
  }

  if (m_db != NULL && m_insert_stmt != NULL)
  {
    int res = SQLITE_OK;

    // Bind the statement parameters.
    res = sqlite3_bind_text(m_insert_stmt, 1, app_id, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'app_id': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 2, timestamp, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'timestamp': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 3, log_stream, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'log_stream': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 4, log_level, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'log_level': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 5, filename, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'filename': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_int(m_insert_stmt, 6, line);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'lin': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 7, class_name, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'class_name': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 8, object_name, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'object_name': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 9, function_name, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'function_name': "
                << sqlite3_errmsg(m_db) << std::endl;
    }
    res = sqlite3_bind_text(m_insert_stmt, 10, message_text, -1, SQLITE_TRANSIENT);
    if (res != SQLITE_OK)
    {
      std::cerr << "SQLite log output: Could not bind column 'message': "
                << sqlite3_errmsg(m_db) << std::endl;
    }

    // Execute the statement.
    res = sqlite3_step(m_insert_stmt);
    if (res != SQLITE_DONE)
    {
      std::cerr << "SQLite log output: Could not insert log line: "
                << sqlite3_errmsg(m_db) << std::endl;
    }

    // Reset the prepared statement.
    sqlite3_reset(m_insert_stmt);
  }
}

}
}
