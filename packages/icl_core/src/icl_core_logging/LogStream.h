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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 * \brief   Contains icl_logging::LogStream
 *
 * \b icl_logging::LogStream
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOG_STREAM_H_INCLUDED
#define ICL_CORE_LOGGING_LOG_STREAM_H_INCLUDED


#include <list>
#include <map>
#include <set>
#include <string>

#include "icl_core/BaseTypes.h"
#include "icl_core/Noncopyable.h"
#include "icl_core/os_thread.h"
#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogLevel.h"
#include "icl_core_logging/Semaphore.h"

namespace icl_core {
namespace logging {

class LogOutputStream;
class LogStream;
class ThreadStream;

typedef std::set<LogStream*> LogStreamSet;

//! Implements a thread-safe logging framework.
class ICL_CORE_LOGGING_IMPORT_EXPORT LogStream : private icl_core::Noncopyable
{
  friend class LoggingManager;
  friend class ThreadStream;

public:
  /*! Creates a new logstream which is not yet connected to any log
   *  output stream.
   */
  LogStream(const icl_core::String& name, icl_core::logging::LogLevel initial_level = cDEFAULT_LOG_LEVEL);

  ~LogStream();

  //! Return the name of the log stream.
  icl_core::String name() const { return m_name; }
  const char *nameCStr() const { return m_name.c_str(); }

  //! Activates or deactivates the log stream.
  void setActive(bool active = true) { m_active = active; }

  //! Get the initial log level of this log stream.
  icl_core::logging::LogLevel initialLogLevel() const { return m_initial_level; }

  //! Checks if the log stream is active.
  bool isActive() const { return m_active; }

  //! Get the log level of the current thread.
  icl_core::logging::LogLevel getLogLevel() const;

  /*! Adds a new log output stream to the log stream.  All log
   *  messages are additionally written to this new log output stream.
   */
  void addOutputStream(LogOutputStream *new_stream);

  /*! Removes a log output stream from the log stream.  Log messages
   *  are no longer written to this log output stream.
   */
  void removeOutputStream(LogOutputStream *stream);

  /*! Returns the underlying thread stream for the current thread.
   *
   *  This function should usually not be used directly.  It is mainly
   *  intended to be used indirectly via the LOGGING_* log macros.
   */
  icl_core::logging::ThreadStream& threadStream(icl_core::logging::LogLevel log_level);

  //! Prints the list of connected log output streams.
  void printConfiguration() const;

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Return the name of the log stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Name() const ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE const char *NameCStr() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Activates or deactivates the log stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetActive(bool active = true) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Get the initial log level of this log stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::logging::LogLevel InitialLogLevel() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Checks if the log stream is active.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool IsActive() const ICL_CORE_GCC_DEPRECATE_STYLE;

  ICL_CORE_VC_DEPRECATE_STYLE_USE(LogStream::getLogLevel)
  icl_core::logging::LogLevel LogLevel() const ICL_CORE_GCC_DEPRECATE_STYLE_USE(LogStream::getLogLevel);

  /*! Adds a new log output stream to the log stream.  All log
   *  messages are additionally written to this new log output stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AddOutputStream(LogOutputStream *new_stream) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Removes a log output stream from the log stream.  Log messages
   *  are no longer written to this log output stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RemoveOutputStream(LogOutputStream *stream) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Prints the list of connected log output streams.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void PrintConfiguration() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the underlying thread stream for the current thread.
   *
   *  This function should usually not be used directly.  It is mainly
   *  intended to be used indirectly via the LOGGING_* log macros.
   */
  ICL_CORE_VC_DEPRECATE_STYLE
    icl_core::logging::ThreadStream& ThreadStream(icl_core::logging::LogLevel log_level)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

private:
  void releaseThreadStream(icl_core::logging::ThreadStream *thread_stream);

  icl_core::logging::LogLevel m_initial_level;

  struct ThreadStreamInfo
  {
    ThreadStreamInfo(ThreadId thread_id,
                     icl_core::logging::LogLevel log_level,
                     icl_core::logging::ThreadStream *thread_stream)
      : thread_id(thread_id),
        log_level(log_level),
        thread_stream(thread_stream)
    { }

    ThreadId thread_id;
    icl_core::logging::LogLevel log_level;
    icl_core::logging::ThreadStream *thread_stream;
  };
  typedef std::list<ThreadStreamInfo> ThreadStreamMap;
  ThreadStreamMap m_thread_stream_map;

  icl_core::String m_name;
  bool m_active;
  std::set<LogOutputStream*> m_output_stream_list;

  // Safe access to the output stream list.
  Semaphore m_mutex;

  static ThreadId m_empty_thread_id;
};

}
}

#endif
