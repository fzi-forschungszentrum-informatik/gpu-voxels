// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 */
//----------------------------------------------------------------------
#include "LogOutputStream.h"

#include <assert.h>
#include <cctype>
#include <cstring>
#include <iostream>

#include <icl_core/internal_raw_debug.h>
#include <icl_core/os.h>
#include <icl_core/os_lxrt.h>
#include <icl_core/os_string.h>
#include <icl_core_config/Config.h>

#include "LoggingManager.h"
#include "ThreadStream.h"

namespace icl_core {
namespace logging {

const icl_core::String LogOutputStream::m_default_log_format = "<~T.~3M> ~S(~L)~ C~(O~::D: ~E";
const icl_core::ThreadPriority LogOutputStream::m_default_worker_thread_priority = 5;

LogOutputStream::LogOutputStream(const icl_core::String& name,
                                 const icl_core::String& config_prefix,
                                 icl_core::logging::LogLevel log_level,
                                 bool use_worker_thread)
  : m_name(name),
    m_log_level(log_level),
    m_time_format("%Y-%m-%d %H:%M:%S"),
    m_use_worker_thread(use_worker_thread),
    m_no_worker_thread_push_mutex(1),
    m_format_mutex(1)
{
  LoggingManager::instance().assertInitialized();

  icl_core::String log_format = m_default_log_format;
  icl_core::config::get<icl_core::String>(config_prefix + "/Format", log_format);
  changeLogFormat(log_format.c_str());

  if (m_use_worker_thread)
  {
    icl_core::ThreadPriority priority = m_default_worker_thread_priority;
    icl_core::config::get<icl_core::ThreadPriority>(config_prefix + "/ThreadPriority", priority);

#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
    size_t message_queue_size = cDEFAULT_FIXED_OUTPUT_STREAM_QUEUE_SIZE;
    icl_core::config::get<size_t>(config_prefix + "/MesageQueueSize", message_queue_size);

    m_worker_thread = new WorkerThread(this, message_queue_size, priority);
#else
    m_worker_thread = new WorkerThread(this, priority);
#endif
  }
  else
  {
    m_worker_thread = NULL;
  }
}

LogOutputStream::LogOutputStream(const icl_core::String& name,
                                 icl_core::logging::LogLevel log_level,
                                 bool use_worker_thread)
  : m_name(name),
    m_log_level(log_level),
    m_time_format("%Y-%m-%d %H:%M:%S"),
    m_use_worker_thread(use_worker_thread),
    m_no_worker_thread_push_mutex(1),
    m_format_mutex(1)
{
  LoggingManager::instance().assertInitialized();
  changeLogFormat(m_default_log_format.c_str());
  if (m_use_worker_thread)
  {
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
    m_worker_thread = new WorkerThread(this, cDEFAULT_FIXED_OUTPUT_STREAM_QUEUE_SIZE,
                                       m_default_worker_thread_priority);
#else
    m_worker_thread = new WorkerThread(this, m_default_worker_thread_priority);
#endif
  }
  else
  {
    m_worker_thread = NULL;
  }
}

LogOutputStream::~LogOutputStream()
{
  if (m_use_worker_thread)
  {
    if (m_worker_thread->running())
    {
      std::cerr << "WARNING: Destroyed LogOutputStream while thread is still alive. "
                << "Please call Shutdown() before destruction." << std::endl;
    }

    delete m_worker_thread;
    m_worker_thread = NULL;
  }
}

void LogOutputStream::changeLogFormat(const char *format)
{
  // Stop processing at the end of the format string.
  if (format[0] != 0)
  {
    parseLogFormat(format);

    if (m_format_mutex.wait())
    {
      m_log_format.clear();
      m_log_format = m_new_log_format;
      m_new_log_format.clear();

      m_format_mutex.post();
    }
  }
}

void LogOutputStream::push(icl_core::logging::LogLevel log_level,
                           const char* log_stream_description, const char *filename,
                           int line, const char *classname, const char *objectname,
                           const char *function, const char *text)
{
  if (log_level >= getLogLevel())
  {
    LogMessage new_entry(icl_core::TimeStamp::now(), log_level, log_stream_description,
                         filename, line, classname, objectname, function, text);

    if (m_use_worker_thread)
    {
      // Hand the log text over to the output implementation.
      if (m_worker_thread->m_mutex->wait())
      {
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
        if (!m_worker_thread->isMessageQueueFull())
        {
          m_worker_thread->m_message_queue[m_worker_thread->m_message_queue_write_index] = new_entry;
          m_worker_thread->incrementIndex(m_worker_thread->m_message_queue_write_index);
        }
#else
        m_worker_thread->m_message_queue.push(new_entry);
#endif
        m_worker_thread->m_mutex->post();
        m_worker_thread->m_fill_count->post();
      }
    }
    else
    {
      if (m_no_worker_thread_push_mutex.wait())
      {
        pushImpl(new_entry);
        m_no_worker_thread_push_mutex.post();
      }
    }
  }
}

void LogOutputStream::pushImpl(const LogMessage& log_message)
{
  if (m_format_mutex.wait())
  {
    std::stringstream msg;
    for (icl_core::List<LogFormatEntry>::const_iterator it = m_log_format.begin();
         it != m_log_format.end(); ++it)
    {
      switch (it->type)
      {
      case LogFormatEntry::eT_TEXT:
      {
        msg << it->text;
        break;
      }
      case LogFormatEntry::eT_CLASSNAME:
      {
        if (std::strcmp(log_message.class_name, "") != 0)
        {
          msg << it->text << log_message.class_name;
        }
        break;
      }
      case LogFormatEntry::eT_OBJECTNAME:
      {
        if (std::strcmp(log_message.object_name, "") != 0)
        {
          msg << it->text << log_message.object_name << it->suffix;
        }
        break;
      }
      case LogFormatEntry::eT_FUNCTION:
      {
        if (std::strcmp(log_message.function_name, "") != 0)
        {
          msg << it->text << log_message.function_name;
        }
        break;
      }
      case LogFormatEntry::eT_MESSAGE:
      {
        msg << log_message.message_text;
        break;
      }
      case LogFormatEntry::eT_FILENAME:
      {
        msg << log_message.filename;
        break;
      }
      case LogFormatEntry::eT_LINE:
      {
        msg << log_message.line;
        break;
      }
      case LogFormatEntry::eT_LEVEL:
      {
        msg << logLevelDescription(log_message.log_level);
        break;
      }
      case LogFormatEntry::eT_STREAM:
      {
        msg << log_message.log_stream;
        break;
      }
      case LogFormatEntry::eT_TIMESTAMP:
      {
        char time_buffer[100];
        memset(time_buffer, 0, 100);

#ifdef _SYSTEM_LXRT_
        // Don't use strfLocaltime() in a hard realtime LXRT task, because
        // it might use a POSIX mutex!
        if (icl_core::os::isThisLxrtTask() && icl_core::os::isThisHRT())
        {
          icl_core::os::snprintf(time_buffer, 99, "%d %02d:%02d:%02d(HRT)",
                                 int(log_message.timestamp.days()),
                                 int(log_message.timestamp.hours()),
                                 int(log_message.timestamp.minutes()),
                                 int(log_message.timestamp.seconds()));
        }
        else
#endif
        {
          log_message.timestamp.strfLocaltime(time_buffer, 100, m_time_format);
        }

        msg << time_buffer;
        break;
      }
      case LogFormatEntry::eT_TIMESTAMP_MS:
      {
        int32_t msec = log_message.timestamp.tsNSec() / 1000000;
        size_t msec_len = 1;
        if (msec >= 10)
        {
          msec_len = 2;
        }
        if (msec >= 100)
        {
          msec_len = 3;
        }
        for (size_t i = it->width; i > msec_len; --i)
        {
          msg << "0";
        }
        msg << msec;
        break;
      }
      }
    }
    m_format_mutex.post();

    pushImpl(msg.str());
  }
}

void LogOutputStream::pushImpl(const icl_core::String&)
{
  std::cerr << "LOG OUTPUT STREAM ERROR: pushImpl() is not implemented!!!" << std::endl;
}

void LogOutputStream::printConfiguration() const
{
  std::cerr << "    " << name() << " : " << logLevelDescription(m_log_level) << std::endl;
}


////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Changes the format of the displayed log timestamp.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::ChangeTimeFormat(const char* format)
{
  changeTimeFormat(format);
}

/*! Change the format of the displayed log entry.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::ChangeLogFormat(const char *format)
{
  changeLogFormat(format);
}

/*! Pushes log data to the log output stream.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::Push(icl_core::logging::LogLevel log_level,
                           const char *log_stream_description, const char *filename,
                           int line, const char *classname, const char *objectname,
                           const char *function, const char *text)
{
  push(log_level, log_stream_description, filename, line, classname, objectname, function, text);
}

/*! Starts the worker thread of the log output stream.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::Start()
{
  start();
}

/*! Shuts down the log output stream. Waits until the logging thread
 *  has finished.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::Shutdown()
{
  shutdown();
}

/*! Return the current log level of this thread stream.
 *  \deprecated Obsolete coding style.
 */
ICL_CORE_VC_DEPRECATE_STYLE_USE(LogOutputStream::getLogLevel)
icl_core::logging::LogLevel LogOutputStream::LogLevel() const
{
  return getLogLevel();
}

/*! Sets the log level of this output stream.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::SetLogLevel(icl_core::logging::LogLevel log_level)
{
  setLogLevel(log_level);
}

/*! Returns the name of this log output stream.
 *  \deprecated Obsolete coding style.
 */
icl_core::String LogOutputStream::Name() const
{
  return name();
}

/*! Prints the configuration (i.e. name, argument and log level) of
 *  this log output stream to cerr.
 *  \deprecated Obsolete coding style.
 */
void LogOutputStream::PrintConfiguration() const
{
  printConfiguration();
}

#endif
/////////////////////////////////////////////////

void LogOutputStream::parseLogFormat(const char *format)
{
  LogFormatEntry new_entry;

  // The format string starts with a field specifier.
  if (format[0] == '~')
  {
    ++format;

    // Read the field width.
    while (format[0] != 0 && std::isdigit(format[0]))
    {
      new_entry.width = 10 * new_entry.width + (format[0] - '0');
      ++format;
    }

    // Read optional prefix text.
    char *prefix_ptr = new_entry.text;
    while (format[0] != 0 && format[0] != 'C' && format[0] != 'O' && format[0] != 'D'
           && format[0] != 'E' && format[0] != 'F' && format[0] != 'G' && format[0] != 'L'
           && format[0] != 'S' && format[0] != 'T' && format[0] != 'M')
    {
      *prefix_ptr = format[0];
      ++prefix_ptr;
      ++format;
    }

    // Read the field type.
    if (format[0] == 'C')
    {
      new_entry.type = LogFormatEntry::eT_CLASSNAME;
    }
    else if (format[0] == 'O')
    {
      new_entry.type = LogFormatEntry::eT_OBJECTNAME;
      if (new_entry.text[0] == '(')
      {
        std::strncpy(new_entry.suffix, ")", 100);
      }
      else if (new_entry.text[0] == '[')
      {
        std::strncpy(new_entry.suffix, "]", 100);
      }
      else if (new_entry.text[0] == '{')
      {
        std::strncpy(new_entry.suffix, "}", 100);
      }
    }
    else if (format[0] == 'D')
    {
      new_entry.type = LogFormatEntry::eT_FUNCTION;
    }
    else if (format[0] == 'E')
    {
      new_entry.type = LogFormatEntry::eT_MESSAGE;
    }
    else if (format[0] == 'F')
    {
      new_entry.type = LogFormatEntry::eT_FILENAME;
    }
    else if (format[0] == 'G')
    {
      new_entry.type = LogFormatEntry::eT_LINE;
    }
    else if (format[0] == 'L')
    {
      new_entry.type = LogFormatEntry::eT_LEVEL;
    }
    else if (format[0] == 'S')
    {
      new_entry.type = LogFormatEntry::eT_STREAM;
    }
    else if (format[0] == 'T')
    {
      new_entry.type = LogFormatEntry::eT_TIMESTAMP;
    }
    else if (format[0] == 'M')
    {
      new_entry.type = LogFormatEntry::eT_TIMESTAMP_MS;
    }

    if (format[0] != 0)
    {
      m_new_log_format.push_back(new_entry);
    }

    ++format;
  }
  else
  {
    char *text_ptr = new_entry.text;
    while (format[0] != '~' && format[0] != 0)
    {
      *text_ptr = format[0];
      ++text_ptr;
      ++format;
    }

    if (new_entry.text[0] != 0)
    {
      m_new_log_format.push_back(new_entry);
    }
  }

  // Stop processing at the end of the format string.
  if (format[0] == 0)
  {
    return;
  }
  else
  {
    parseLogFormat(format);
  }
}

void LogOutputStream::start()
{
  if (m_use_worker_thread)
  {
    m_worker_thread->start();
  }
}

void LogOutputStream::shutdown()
{
  if (m_use_worker_thread)
  {
    if (m_worker_thread->running())
    {
      m_worker_thread->stop();
      m_worker_thread->m_fill_count->post();
      m_worker_thread->join();
    }
  }
}

#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
LogOutputStream::WorkerThread::WorkerThread(LogOutputStream *output_stream, size_t message_queue_size,
                                            icl_core::ThreadPriority priority)
  : Thread(priority),
    m_output_stream(output_stream),
    m_message_queue_size(message_queue_size),
    m_message_queue_write_index(0),
    m_message_queue_read_index(0)
{
  m_message_queue = new LogMessage[message_queue_size+1];

  m_mutex = new Semaphore(1);
  m_fill_count = new Semaphore(0);
}
#else
LogOutputStream::WorkerThread::WorkerThread(LogOutputStream *output_stream, icl_core::ThreadPriority priority)
  : Thread(priority),
    m_output_stream(output_stream)
{
  m_mutex = new Semaphore(1);
  m_fill_count = new Semaphore(0);
}
#endif

LogOutputStream::WorkerThread::~WorkerThread()
{
  delete m_mutex;
  delete m_fill_count;
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
  delete[] m_message_queue;
#endif
}

void LogOutputStream::WorkerThread::run()
{
  m_output_stream->onStart();

  // Wait for new messages to arrive.
  while (execute())
  {
    if (m_fill_count->wait())
    {
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
      if (!isMessageQueueEmpty())
#else
      if (!m_message_queue.empty())
#endif
      {
        if (m_mutex->wait())
        {
          LogMessage log_message;
          bool push = false;
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
          if (!isMessageQueueEmpty())
          {
            log_message = m_message_queue[m_message_queue_read_index];
            incrementIndex(m_message_queue_read_index);
            push = true;
          }
#else
          if (!m_message_queue.empty())
          {
            log_message = m_message_queue.front();
            m_message_queue.pop();
            push = true;
          }
#endif

          m_mutex->post();

          if (push)
          {
            m_output_stream->pushImpl(log_message);
          }
        }
      }
    }
    else if (execute())
    {
      PRINTF("LogOutputStream(%s)::run: semaphore wait failed\n", m_output_stream->m_name.c_str());
      icl_core::os::usleep(10000);
    }
  }

  // Write out all remaining log messages.
  if (m_mutex->wait())
  {
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
    while (!isMessageQueueEmpty())
    {
      LogMessage log_message = m_message_queue[m_message_queue_read_index];
      incrementIndex(m_message_queue_read_index);
      m_output_stream->pushImpl(log_message);
    }
#else
    while (!m_message_queue.empty())
    {
      LogMessage log_message = m_message_queue.front();
      m_message_queue.pop();
      m_output_stream->pushImpl(log_message);
    }
#endif

    m_mutex->post();
  }

  m_output_stream->onShutdown();
}


#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
void LogOutputStream::WorkerThread::incrementIndex(size_t& index)
{
  ++index;
  if (index >= m_message_queue_size)
  {
    index = 0;
  }
}

bool LogOutputStream::WorkerThread::isMessageQueueEmpty()
{
  return m_message_queue_read_index == m_message_queue_write_index;
}

bool LogOutputStream::WorkerThread::isMessageQueueFull()
{
  return ((m_message_queue_write_index == m_message_queue_read_index - 1)
          || (m_message_queue_write_index == m_message_queue_size - 1
              && m_message_queue_read_index == 0));
}
#endif

LogOutputStream::LogMessage::LogMessage(const icl_core::TimeStamp& timestamp,
                                        icl_core::logging::LogLevel log_level,
                                        const char *log_stream, const char *filename,
                                        size_t line, const char *class_name,
                                        const char *object_name, const char *function_name,
                                        const char *message_text)
  : timestamp(timestamp),
    log_level(log_level),
    line(line)
{
  std::strncpy(LogMessage::log_stream, log_stream, cMAX_IDENTIFIER_LENGTH+1);
  std::strncpy(LogMessage::filename, filename, cMAX_DESCRIPTION_LENGTH+1);
  std::strncpy(LogMessage::class_name, class_name, cMAX_IDENTIFIER_LENGTH+1);
  std::strncpy(LogMessage::object_name, object_name, cMAX_DESCRIPTION_LENGTH+1);
  std::strncpy(LogMessage::function_name, function_name, cMAX_IDENTIFIER_LENGTH+1);
  std::strncpy(LogMessage::message_text, message_text, cDEFAULT_LOG_SIZE+1);
}

}
}
