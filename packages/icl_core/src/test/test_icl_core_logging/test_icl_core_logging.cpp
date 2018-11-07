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
 * \date    2008-04-08
 *
 */
//----------------------------------------------------------------------
#include <icl_core/BaseTypes.h>
#include <icl_core/internal_raw_debug.h>
#include <icl_core/os_lxrt.h>
#include <icl_core_config/Config.h>
#include <icl_core_logging/Logging.h>
#include <icl_core_logging/ScopedTimer.h>
#include <icl_core_thread/Thread.h>

DECLARE_LOG_STREAM(Main);
DECLARE_LOG_STREAM(ThreadLog);

REGISTER_LOG_STREAM(Main)
REGISTER_LOG_STREAM(ThreadLog)

using icl_core::logging::endl;

class LoggingThread : public icl_core::thread::Thread
{
public:
  LoggingThread(icl_core::ThreadPriority priority,
                size_t message_count,
                uint32_t sleep_time_us)
    : icl_core::thread::Thread("LoggingThread", priority),
      m_message_count(message_count),
      m_sleep_time_us(sleep_time_us)
  { }

  virtual void run()
  {
#ifdef _SYSTEM_LXRT_
    PRINTF("test_icl_core_logging(LoggingThread): If you see this on the console then I am not hard real-time!\n");
#endif

    LOGGING_ERROR_C(ThreadLog, LoggingThread, "Thread Error" << endl);
    LOGGING_WARNING_C(ThreadLog, LoggingThread, "Thread Warning" << endl);
    LOGGING_INFO_C(ThreadLog, LoggingThread, "Thread Info" << endl);
    LOGGING_DEBUG_C(ThreadLog, LoggingThread, "Thread Debug" << endl);
    LOGGING_TRACE_C(ThreadLog, LoggingThread, "Thread Trace" << endl);

    for (size_t i = 0; i < m_message_count; ++i)
    {
      LOGGING_INFO_C(ThreadLog, LoggingThread, "Thread Loop " << i << endl);
      if (m_sleep_time_us > 0)
      {
        icl_core::os::usleep(m_sleep_time_us);
      }
    }
  }

private:
  size_t m_message_count;
  uint32_t m_sleep_time_us;
};

int main(int argc, char *argv[])
{
  icl_core::os::lxrtStartup();

  icl_core::config::addParameter(icl_core::config::ConfigParameter("priority:", "p", "/TestLogging/ThreadPriority", "Priority of the logging thread."));
  icl_core::config::addParameter(icl_core::config::ConfigParameter("message-count:", "c", "/TestLogging/MessageCount", "Number of messages to be logged."));
  icl_core::config::addParameter(icl_core::config::ConfigParameter("sleep-time:", "s", "/TestLogging/SleepTimeUS", "Sleep time (us) between two log messages."));

  icl_core::logging::initialize(argc, argv);

  icl_core::ThreadPriority priority = icl_core::config::getDefault<icl_core::ThreadPriority>("/TestLogging/ThreadPriority", 0);
  size_t message_count = icl_core::config::getDefault<size_t>("/TestLogging/MessageCount", 10);
  uint32_t sleep_time_us = icl_core::config::getDefault<uint32_t>("/TestLogging/SleepTimeUS", 1000);

  LOGGING_INFO(Main, "Creating logging thread." << endl);
  LoggingThread *thread = new LoggingThread(priority, message_count, sleep_time_us);
  LOGGING_INFO(Main, "Starting logging thread." << endl);
  thread->start();

  {
    LOGGING_SCOPED_TIMER_INFO(Main, "Main Loop");
    for (size_t i = 0; i < message_count; ++i)
    {
      LOGGING_INFO(Main, "Main Loop " << i << endl);
      if (sleep_time_us > 0)
      {
        icl_core::os::usleep(sleep_time_us);
      }
    }
  }

  LOGGING_INFO(Main, "Waiting for logging thread to finish." << endl);
  thread->join();
  LOGGING_INFO(Main, "Logging thread finished." << endl);
  delete thread;

  icl_core::logging::LoggingManager::instance().shutdown();
  icl_core::os::lxrtShutdown();

  return 0;
}
