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
 * \date    2006-05-10
 *
 * \brief   Contains icl_logging::LogOutputStream
 *
 * \b icl_logging::LogOutputStream
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOG_OUTPUT_STREAM_H_INCLUDED
#define ICL_CORE_LOGGING_LOG_OUTPUT_STREAM_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/os_mem.h>
#include <icl_core/List.h>
#include <icl_core/Noncopyable.h>
#include <icl_core/TimeStamp.h>

#ifdef _SYSTEM_LXRT_
#define ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
#else
# include <icl_core/Queue.h>
#endif

#include "icl_core_logging/Constants.h"
#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LoggingManager.h"
#include "icl_core_logging/LogLevel.h"
#include "icl_core_logging/Semaphore.h"
#include "icl_core_logging/Thread.h"

namespace icl_core {
namespace logging {

/*! \brief This is an output stream class for log messages.
 *
 *  LogOutputStream is responsible for actually outputting log
 *  messages. This class itself is purely virtual so only derived
 *  classes which implement the virtual pushImpl() function can be
 *  used.
 *
 *  Child classes of LogOutputStream can implement any suitable kind
 *  of logging output, e.g. console output (i.e. stdout), console
 *  error (i.e. stderr), file output or even a client for a networked
 *  logging server.
 *
 *  By using different log output streams and registering them with
 *  selected log streams one can control which logging information is
 *  logged to which log output. This way one can e.g. log different
 *  kinds of log messages to different files and additionally output
 *  everything onto the console.
 *
 *  The implementations tStdLogOutput, tStdErrLogOutput and
 *  tFileLogOutput are provided by this library and can be used out of
 *  the box.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT LogOutputStream : protected virtual icl_core::Noncopyable
{
public:
  /*! Creates a new log output stream.
   *
   *  \param name Name of the log output stream.
   *  \param config_prefix Config prefix for output format
   *         configuration.
   *  \param log_level Sets the initial log level of this output
   *         stream.
   *  \param use_worker_thread If \c true, creates a separate worker
   *         thread which actually outputs the log data.
   */
  LogOutputStream(const icl_core::String& name, const icl_core::String& config_prefix,
                  icl_core::logging::LogLevel log_level, bool use_worker_thread = true);

  virtual ~LogOutputStream();

  /*! Changes the format of the displayed log timestamp.
   *
   *  The default time format is "%Y-%m-%d %H:%M:%S".
   */
  void changeTimeFormat(const char* format) { m_time_format = format; }

  /*! Change the format of the displayed log entry.
   *
   *  \li <tt>~[n]C</tt>: The name of the class in which the log
   *      message has been generated.  If the classname is empty this
   *      field is omitted.
   *  \li <tt>~[n][br]O</tt>: The name of the object of a class.
   *  \li <tt>~[n][str]D</tt>: The name of the function in which the
   *      log message has been generated.  If the function is empty
   *      this field is omitted.
   *  \li <tt>~E</tt>: The actual log entry text.
   *  \li <tt>~[n]F</tt>: The name of the file in which the log
   *      message has been generated.
   *  \li <tt>~[n]G</tt>: The line number in which the log message has
   *      been generated.
   *  \li <tt>~[n]L</tt>: The log level of the log entry.
   *  \li <tt>~[n]S</tt>: The description of the originating log
   *      stream.
   *  \li <tt>~[n]T</tt>: The timestamp (essentially formatted by
   *      strftime).
   *  \li <tt>~[n]M</tt>: The millisecond part of the timestamp.
   *
   *  \li <tt>[n]</tt> specifies an optional minimum width of the
   *      output field and can be used to line up output fields.
   *  \li <tt>[str]</tt> specifies an optional string which is printed
   *      in front of the output field if it is present.
   *  \li <tt>[br]</tt> The type of parentheses by which the argument
   *      should be enclosed. One of "(", "[" and "{".
   *
   *  The default log entry format is
   *  <tt>"<~T.~3M> ~S(~L)~ C~(O~::D: ~E"</tt>.
   */
  void changeLogFormat(const char *format);

  /*! Pushes log data to the log output stream.
   *
   *  \param log_level The log level of the originating log stream.
   *  \param log_stream_description The description of the originating
   *         log stream.
   *  \param filename Name of the source file where the log message
   *         originated.
   *  \param line Source file line where the log message originated.
   *  \param classname Name of the class in which the log message
   *         originated.
   *  \param objectname Name of the object that created the log
   *         message.
   *  \param function Name of the function in which the log message
   *         originated.
   *  \param text The actual log string.
   */
  void push(icl_core::logging::LogLevel log_level, const char *log_stream_description,
            const char *filename, int line, const char *classname, const char *objectname,
            const char *function, const char *text);

  //! Starts the worker thread of the log output stream.
  void start();

  /*! Shuts down the log output stream. Waits until the logging thread
   *  has finished.
   */
  void shutdown();

  /*! Returns the current log level of this output stream.
   */
  icl_core::logging::LogLevel getLogLevel() const { return m_log_level; }

  /*! Sets the log level of this output stream.
   */
  void setLogLevel(icl_core::logging::LogLevel log_level) { m_log_level = log_level; }

  /*! Returns the name of this log output stream.
   *
   *  Remark: The name of a log output stream is set by the logging
   *  manager.
   */
  icl_core::String name() const { return m_name; }

  /*! Prints the configuration (i.e. name, argument and log level) of
   *  this log output stream to cerr.
   */
  void printConfiguration() const;

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Changes the format of the displayed log timestamp.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void ChangeTimeFormat(const char* format) ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Change the format of the displayed log entry.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void ChangeLogFormat(const char *format)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Pushes log data to the log output stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Push(icl_core::logging::LogLevel log_level,
                                        const char *log_stream_description, const char *filename,
                                        int line, const char *classname, const char *objectname,
                                        const char *function, const char *text)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Starts the worker thread of the log output stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Start() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Shuts down the log output stream. Waits until the logging thread
   *  has finished.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Shutdown() ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Return the current log level of this thread stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE_USE(LogOutputStream::getLogLevel)
  icl_core::logging::LogLevel LogLevel() const
    ICL_CORE_GCC_DEPRECATE_STYLE_USE(LogOutputStream::getLogLevel);

  /*! Sets the log level of this output stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void SetLogLevel(icl_core::logging::LogLevel log_level)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Returns the name of this log output stream.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::String Name() const ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Prints the configuration (i.e. name, argument and log level) of
   *  this log output stream to cerr.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void PrintConfiguration() const ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
/////////////////////////////////////////////////

protected:
  //! Defines an entry for the message queue.
  struct LogMessage
  {
    LogMessage(const icl_core::TimeStamp& timestamp = icl_core::TimeStamp(),
               icl_core::logging::LogLevel log_level = eLL_MUTE,
               const char *log_stream = "", const char *filename = "",
               size_t line = 0,
               const char *class_name = "", const char *object_name = "", const char *function_name = "",
               const char *message_text = "");

    icl_core::TimeStamp timestamp;
    icl_core::logging::LogLevel log_level;
    char log_stream[cMAX_IDENTIFIER_LENGTH + 1];
    char filename[cMAX_DESCRIPTION_LENGTH + 1];
    size_t line;
    char class_name[cMAX_IDENTIFIER_LENGTH + 1];
    char object_name[cMAX_DESCRIPTION_LENGTH + 1];
    char function_name[cMAX_IDENTIFIER_LENGTH + 1];
    char message_text[cDEFAULT_LOG_SIZE + 1];
  };

  /*! An alternative constructor for internal use only.
   */
  LogOutputStream(const icl_core::String& name, icl_core::logging::LogLevel log_level,
                  bool use_worker_thread = true);

private:
  friend class LoggingManager;

  //! Implements processing the message queue in a separate thread.
  struct WorkerThread : public Thread, protected virtual icl_core::Noncopyable
  {
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
    WorkerThread(LogOutputStream *output_stream, size_t message_queue_size,
                 icl_core::ThreadPriority priority);
#else
    WorkerThread(LogOutputStream *output_stream, icl_core::ThreadPriority priority);
#endif
    ~WorkerThread();

    virtual void run();

#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
    void incrementIndex(size_t& index);
    bool isMessageQueueEmpty();
    bool isMessageQueueFull();
#endif

    LogOutputStream *m_output_stream;

#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
    LogMessage *m_message_queue;
    size_t m_message_queue_size;
    size_t m_message_queue_read_index;
    size_t m_message_queue_write_index;
#else
    typedef icl_core::Queue<LogMessage> MessageQueue;
    MessageQueue m_message_queue;
#endif

    Semaphore *m_mutex;
    Semaphore *m_fill_count;
  };
  friend struct WorkerThread;

  /*!
   * Represents an entry in the log format.
   */
  struct LogFormatEntry
  {
    enum EntryType
    {
      eT_TEXT,
      eT_CLASSNAME,
      eT_OBJECTNAME,
      eT_FUNCTION,
      eT_MESSAGE,
      eT_FILENAME,
      eT_LINE,
      eT_LEVEL,
      eT_STREAM,
      eT_TIMESTAMP,
      eT_TIMESTAMP_MS
    };

    LogFormatEntry()
      : type(eT_TEXT), width(0)
    {
      icl_core::os::memset(text, 0, 100);
      icl_core::os::memset(suffix, 0, 100);
    }

    EntryType type;
    size_t width;
    char text[100];
    char suffix[100];
  };

  /*! This virtual function is called from the worker thread just
   *  after it has been started. It can be used by output stream
   *  implementations to do initializations, which have to be
   *  performed in the worker thread.
   */
  virtual void onStart() { }
  /*! This virtual function is called with an unformatted log message.
   *  It can be overridden by output streams, which need to process
   *  the individual fields of a log message.<br/> The default
   *  implementation formats the log message according to the
   *  configured message format and calls pushImpl(const char*) for
   *  further processing.
   */
  virtual void pushImpl(const LogMessage& log_message);
  /*! This virtual function is called with a formatted log line.  It
   *  sould be overridden by output stream implementations, which do
   *  not need the individual fields of a log message but only a
   *  formatted log message text.
   */
  virtual void pushImpl(const std::string& log_line);
  /*! This virtual function is called from the worker thread just
   *  before it ends execution. It can be used by output stream
   *  implementations to do cleanup work, which has to be performed in
   *  the worker thread.
   */
  virtual void onShutdown() { }

  void parseLogFormat(const char *format);

  icl_core::String m_name;
  icl_core::logging::LogLevel m_log_level;
  const char *m_time_format;

  bool m_use_worker_thread;
  WorkerThread *m_worker_thread;
  Semaphore m_no_worker_thread_push_mutex;

  Semaphore m_format_mutex;
  icl_core::List<LogFormatEntry> m_log_format;
  icl_core::List<LogFormatEntry> m_new_log_format;

  static const icl_core::String m_default_log_format;
  static const icl_core::ThreadPriority m_default_worker_thread_priority;
};

}
}

#endif
