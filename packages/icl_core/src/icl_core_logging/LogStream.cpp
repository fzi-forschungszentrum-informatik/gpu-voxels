// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------
#include "LogStream.h"

#include <assert.h>
#include <iostream>

#include "icl_core/os_lxrt.h"
#include "icl_core/os_string.h"
#include "icl_core_logging/LoggingManager.h"
#include "icl_core_logging/LogOutputStream.h"
#include "icl_core_logging/ThreadStream.h"

namespace icl_core {
namespace logging {

ThreadId LogStream::m_empty_thread_id(0);

LogStream::LogStream(const std::string& name, icl_core::logging::LogLevel initial_level)
  : m_initial_level(initial_level),
    m_name(name),
    m_active(true),
    m_mutex(1)
{
  LoggingManager::instance().assertInitialized();

  for (size_t i = 0; i < cDEFAULT_LOG_THREAD_STREAM_POOL_SIZE; ++i)
  {
    m_thread_stream_map.push_back(ThreadStreamInfo(m_empty_thread_id, eLL_MUTE,
                                                   new icl_core::logging::ThreadStream(this)));
  }
}

LogStream::~LogStream()
{
  LoggingManager::instance().removeLogStream(m_name);

  ThreadStreamMap::const_iterator it = m_thread_stream_map.begin();
  for (; it != m_thread_stream_map.end(); ++it)
  {
    delete it->thread_stream;
  }
  m_thread_stream_map.clear();
}

icl_core::logging::LogLevel LogStream::getLogLevel() const
{
  // TODO: Implement individual log levels for each thread.
  return m_initial_level;
}

void LogStream::addOutputStream(LogOutputStream *new_stream)
{
  if (m_mutex.wait())
  {
    m_output_stream_list.insert(new_stream);
    m_mutex.post();
  }
}

void LogStream::removeOutputStream(LogOutputStream *stream)
{
  if (m_mutex.wait())
  {
    m_output_stream_list.erase(stream);
    m_mutex.post();
  }
}

icl_core::logging::ThreadStream& LogStream::threadStream(icl_core::logging::LogLevel log_level)
{
  icl_core::logging::ThreadStream *thread_stream = NULL;

  while (!m_mutex.wait())
  { }

  ThreadId thread_id = icl_core::os::threadSelf();

  // Try to find the stream for the current thread, if it has already been assigned.
  for (ThreadStreamMap::const_iterator find_it = m_thread_stream_map.begin();
       find_it != m_thread_stream_map.end(); ++find_it)
  {
    if (find_it->thread_id == thread_id && find_it->log_level == log_level)
    {
      thread_stream = find_it->thread_stream;
      break;
    }
  }

  // Take a thread stream from the pool, if one is available.
  if (thread_stream == NULL)
  {
    for (ThreadStreamMap::iterator find_it = m_thread_stream_map.begin();
         find_it != m_thread_stream_map.end(); ++find_it)
    {
      if (find_it->thread_id == m_empty_thread_id)
      {
        find_it->thread_id = thread_id;
        find_it->log_level = log_level;
        thread_stream = find_it->thread_stream;
        break;
      }
    }
  }

  // There are no more threads streams available, so create a new one.
  if (thread_stream == NULL)
  {
#ifdef _SYSTEM_LXRT_
    // Leave hard real-time to create a new thread stream.
    bool re_enter_hrt = false;
    if (icl_core::os::isThisHRT())
    {
      icl_core::os::ensureNoHRT();
      re_enter_hrt = true;
    }
#endif
    thread_stream = new icl_core::logging::ThreadStream(this);
    m_thread_stream_map.push_back(ThreadStreamInfo(thread_id, log_level, thread_stream));
#ifdef _SYSTEM_LXRT_
    if (re_enter_hrt)
    {
      icl_core::os::makeHRT();
    }
#endif
  }

  m_mutex.post();

  // Set the log level for the thread stream.
  thread_stream->changeLevel(this->getLogLevel());

  return *thread_stream;
}

void LogStream::printConfiguration() const
{
  for (std::set<LogOutputStream*>::const_iterator it = m_output_stream_list.begin();
       it != m_output_stream_list.end(); ++it)
  {
    std::cerr << (*it)->name() << " ";
  }
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Return the name of the log stream.
   *  \deprecated Obsolete coding style.
   */
  icl_core::String LogStream::Name() const
  {
    return name();
  }

  const char *LogStream::NameCStr() const
  {
    return nameCStr();
  }

  /*! Activates or deactivates the log stream.
   *  \deprecated Obsolete coding style.
   */
  void LogStream::SetActive(bool active)
  {
    setActive(active);
  }

  /*! Get the initial log level of this log stream.
   *  \deprecated Obsolete coding style.
   */
  icl_core::logging::LogLevel LogStream::InitialLogLevel() const
  {
    return initialLogLevel();
  }

  /*! Checks if the log stream is active.
   *  \deprecated Obsolete coding style.
   */
  bool LogStream::IsActive() const
  {
    return isActive();
  }

  ICL_CORE_VC_DEPRECATE_STYLE_USE(LogStream::getLogLevel)
  icl_core::logging::LogLevel LogStream::LogLevel() const
  {
    return getLogLevel();
  }

  /*! Adds a new log output stream to the log stream.  All log
   *  messages are additionally written to this new log output stream.
   *  \deprecated Obsolete coding style.
   */
  void LogStream::AddOutputStream(LogOutputStream *new_stream)
  {
    addOutputStream(new_stream);
  }

  /*! Removes a log output stream from the log stream.  Log messages
   *  are no longer written to this log output stream.
   *  \deprecated Obsolete coding style.
   */
  void LogStream::RemoveOutputStream(LogOutputStream *stream)
  {
    removeOutputStream(stream);
  }

  /*! Prints the list of connected log output streams.
   *  \deprecated Obsolete coding style.
   */
  void LogStream::PrintConfiguration() const
  {
    printConfiguration();
  }

  /*! Returns the underlying thread stream for the current thread.
   *
   *  This function should usually not be used directly.  It is mainly
   *  intended to be used indirectly via the LOGGING_* log macros.
   */
  ICL_CORE_VC_DEPRECATE_STYLE
    icl_core::logging::ThreadStream& LogStream::ThreadStream(icl_core::logging::LogLevel log_level)
  {
    return threadStream(log_level);
  }

#endif
/////////////////////////////////////////////////


void LogStream::releaseThreadStream(icl_core::logging::ThreadStream *thread_stream)
{
  for (ThreadStreamMap::iterator find_it = m_thread_stream_map.begin();
       find_it != m_thread_stream_map.end(); ++find_it)
  {
    if (find_it->thread_stream == thread_stream)
    {
      find_it->thread_id = m_empty_thread_id;
      break;
    }
  }
}

}
}
