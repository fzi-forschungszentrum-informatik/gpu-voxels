// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-02
 *
 */
//----------------------------------------------------------------------
#include "icl_core_logging/LoggingManager.h"

#include <assert.h>
#include <iostream>
#include <sstream>
#include <cstdlib>

#include <boost/lexical_cast.hpp>
#include <boost/tuple/tuple.hpp>

#include <icl_core/os_lxrt.h>
#include <icl_core_config/Config.h>
#include "icl_core_logging/FileLogOutput.h"
#include "icl_core_logging/LogStream.h"
#include "icl_core_logging/StdLogOutput.h"
#include "icl_core_config/GetoptParser.h"
#include "icl_core_config/GetoptParameter.h"

namespace icl_core {
namespace logging {

void LoggingManager::initialize()
{
  if (!m_initialized)
  {
    m_initialized = true;

    // Read the log output stream configuration.
    ::icl_core::config::ConfigIterator output_stream_it =
        ::icl_core::config::find("\\/IclCore\\/Logging\\/(OutputStream.*)\\/(.*)");
    while (output_stream_it.next())
    {
      ::icl_core::String entry_name = output_stream_it.matchGroup(1);
      ::icl_core::String value_name = output_stream_it.matchGroup(2);
      if (value_name == "OutputStreamName")
      {
        m_output_stream_config[entry_name].output_stream_name = output_stream_it.value();
      }
      else if (value_name == "Name")
      {
        m_output_stream_config[entry_name].name = output_stream_it.value();
      }
      else if (value_name == "LogLevel")
      {
        if (!stringToLogLevel(output_stream_it.value(), m_output_stream_config[entry_name].log_level))
        {
          std::cerr << "LOGGING CONFIG ERROR: Illegal log level in " << output_stream_it.key() << std::endl;
        }
      }
      else if (value_name.substr(0, 9) == "LogStream")
      {
        m_output_stream_config[entry_name].log_streams.push_back(output_stream_it.value());
      }
    }

    // Read the log stream configuration.
    ::icl_core::config::ConfigIterator log_stream_it =
        ::icl_core::config::find("\\/IclCore\\/Logging\\/(LogStream.*)\\/(.*)");
    while (log_stream_it.next())
    {
      ::icl_core::String entry_name = log_stream_it.matchGroup(1);
      ::icl_core::String value_name = log_stream_it.matchGroup(2);
      if (value_name == "Name")
      {
        m_log_stream_config[entry_name].name = log_stream_it.value();
      }
      else if (value_name == "LogLevel")
      {
        if (!stringToLogLevel(log_stream_it.value(), m_log_stream_config[entry_name].log_level))
        {
          std::cerr << "LOGGING CONFIG ERROR: Illegal log level in " << log_stream_it.key() << std::endl;
        }
      }
    }
  }

  configure();

  // Configure the "QuickDebug" log stream and log output stream.
  icl_core::String quick_debug_filename;
  if (icl_core::config::paramOpt<icl_core::String>("quick-debug", quick_debug_filename))
  {
    // Find an unused name for the QuickDebug[0-9]* log output stream.
    icl_core::String output_stream_name = "QuickDebug";
    LogOutputStreamMap::const_iterator find_it = m_log_output_streams.find(output_stream_name);
    if (find_it != m_log_output_streams.end())
    {
      size_t count = 0;
      do
      {
        ++count;
        find_it = m_log_output_streams.find(output_stream_name
                                            + boost::lexical_cast<icl_core::String>(count));
      }
      while (find_it != m_log_output_streams.end());
      output_stream_name = output_stream_name + boost::lexical_cast<icl_core::String>(count);
    }

    // Create the log output stream and connect the log stream.
    LogOutputStream *output_stream = new FileLogOutput(output_stream_name, quick_debug_filename,
                                                       eLL_TRACE, true);
    m_log_output_streams[output_stream_name] = output_stream;
    QuickDebug::instance().addOutputStream(output_stream);
    QuickDebug::instance().m_initial_level = eLL_TRACE;
  }

  // Run the log output stream threads.
  if (m_default_log_output != 0)
  {
    m_default_log_output->start();
  }
  for (LogOutputStreamMap::iterator output_stream_it = m_log_output_streams.begin();
       output_stream_it != m_log_output_streams.end();
       ++output_stream_it)
  {
    output_stream_it->second->start();
  }
}

void LoggingManager::configure()
{
  // Create the default log output stream, if necessary.
  if (m_output_stream_config.empty() && m_default_log_output == NULL)
  {
    m_default_log_output = StdLogOutput::create("Default", "/IclCore/Logging/Default");
  }

  // Create log stream instances, if necessary.
  for (LogStreamFactoryMap::iterator log_stream_it = m_log_stream_factories.begin();
       log_stream_it != m_log_stream_factories.end(); ++log_stream_it)
  {
    if (m_log_streams.find(log_stream_it->first) == m_log_streams.end())
    {
      registerLogStream((*log_stream_it->second)());
    }
  }

  // Delete the default log output stream, if necessary.
  if (!m_output_stream_config.empty() && m_default_log_output != NULL)
  {
    for (LogStreamMap::iterator it = m_log_streams.begin(); it != m_log_streams.end(); ++it)
    {
      it->second->removeOutputStream(m_default_log_output);
    }

    m_default_log_output->shutdown();
    delete m_default_log_output;
    m_default_log_output = 0;
  }

  // Run through the log output stream configuration
  for (LogOutputStreamConfigMap::iterator loc_it = m_output_stream_config.begin();
       loc_it != m_output_stream_config.end(); ++loc_it)
  {
    // Auto-generate a suitable name for the log output stream, if it
    // has not been set in the configuration.
    if (loc_it->second.name == ::icl_core::String())
    {
      loc_it->second.name = loc_it->second.output_stream_name;
    }

    // Create the configured log output stream, if necessary.
    LogOutputStreamMap::const_iterator find_log_output_stream =
      m_log_output_streams.find(loc_it->second.name);
    if (find_log_output_stream == m_log_output_streams.end())
    {
      LogOutputStreamFactoryMap::const_iterator find_log_output_stream_factory =
        m_log_output_stream_factories.find(loc_it->second.output_stream_name);
      if (find_log_output_stream_factory == m_log_output_stream_factories.end())
      {
        // If the log output stream cannot be created then skip to the
        // next configuration entry.
        continue;
      }
      LogOutputStream *log_output_stream =
        (*find_log_output_stream_factory->second)(loc_it->second.name,
                                                  "/IclCore/Logging/" + loc_it->first,
                                                  loc_it->second.log_level);
      boost::tuples::tie(find_log_output_stream, boost::tuples::ignore) =
        m_log_output_streams.insert(std::make_pair(loc_it->second.name, log_output_stream));
    }

    // Check again, just to be sure!
    if (find_log_output_stream != m_log_output_streams.end())
    {
      // Connect the configured log streams (either the list from the
      // commandline or all available log streams).
      if (loc_it->second.log_streams.empty())
      {
        for (LogStreamMap::iterator it = m_log_streams.begin(); it != m_log_streams.end(); ++it)
        {
          it->second->addOutputStream(find_log_output_stream->second);
        }
      }
      else
      {
        for (StringList::const_iterator it = loc_it->second.log_streams.begin();
             it != loc_it->second.log_streams.end(); ++it)
        {
          LogStreamMap::iterator find_it = m_log_streams.find(*it);
          if (find_it == m_log_streams.end())
          {
            // If the log stream cannot be found then skip to the next
            // entry.  Maybe there will be a second call to configure()
            // in the future and the log stream is available then.
            continue;
          }
          else
          {
            find_it->second->addOutputStream(find_log_output_stream->second);
          }
        }
      }
    }
  }

  // Set the log level of the configured log streams (either the list
  // from the commandline or all available log streams).
  for (LogStreamConfigMap::const_iterator lsc_it = m_log_stream_config.begin();
       lsc_it != m_log_stream_config.end(); ++lsc_it)
  {
    LogStreamMap::iterator find_it = m_log_streams.find(lsc_it->second.name);
    if (find_it == m_log_streams.end())
    {
      // If the log stream cannot be found then skip to the next
      // entry.  Maybe there will be a second call to configure() in
      // the future and the log stream is available then.
      continue;
    }
    else
    {
      find_it->second->m_initial_level = lsc_it->second.log_level;
    }
  }


  if (icl_core::config::Getopt::instance().paramOptPresent("log-level"))
  {
    LogLevel initial_level = cDEFAULT_LOG_LEVEL;
    icl_core::String log_level = icl_core::config::Getopt::instance().paramOpt("log-level");
    if (!stringToLogLevel(log_level, initial_level))
    {
      std::cerr << "Illegal log level " << log_level << std::endl;
      std::cerr << "Valid levels are 'Trace', 'Debug', 'Info', 'Warning', 'Error' and 'Mute'." << std::endl;
    }
    else
    {
      if (m_default_log_output == NULL)
      {
        m_default_log_output = StdLogOutput::create("Default", "/IclCore/Logging/Default");
      }
      m_default_log_output->setLogLevel(initial_level);

      for (LogStreamMap::iterator lsm_it = m_log_streams.begin(); lsm_it != m_log_streams.end(); ++lsm_it)
      {
        lsm_it->second->m_initial_level = initial_level;
        lsm_it->second->addOutputStream(m_default_log_output);
      }

      for (LogOutputStreamMap::iterator los_it = m_log_output_streams.begin(); los_it
             != m_log_output_streams.end(); ++los_it)
      {
        los_it->second->setLogLevel(initial_level);
      }
    }
  }
}

void LoggingManager::assertInitialized() const
{
  if (!initialized())
  {
    assert(0);
  }
}

void LoggingManager::registerLogOutputStream(const ::icl_core::String& name, LogOutputStreamFactory factory)
{
  m_log_output_stream_factories[name] = factory;
}

void LoggingManager::removeLogOutputStream(LogOutputStream *log_output_stream, bool remove_from_list)
{
  for (LogStreamMap::iterator log_stream_it = m_log_streams.begin();
       log_stream_it != m_log_streams.end();
       ++log_stream_it)
  {
    log_stream_it->second->removeOutputStream(log_output_stream);
  }

  if (remove_from_list)
  {
    m_log_output_streams.erase(log_output_stream->name());
  }
}

void LoggingManager::registerLogStream(const icl_core::String& name, LogStreamFactory factory)
{
  m_log_stream_factories[name] = factory;
}

void LoggingManager::registerLogStream(LogStream *log_stream)
{
  m_log_streams[log_stream->name()] = log_stream;

  if (m_default_log_output != 0)
  {
    log_stream->addOutputStream(m_default_log_output);
  }
}

void LoggingManager::removeLogStream(const icl_core::String& log_stream_name)
{
  if (!m_shutdown_running)
  {
    m_log_streams.erase(log_stream_name);
  }
}

LoggingManager::LoggingManager()
{
  m_initialized = false;
  m_shutdown_running = false;
  m_default_log_output = NULL;

  icl_core::String help_text =
    "Override the log level of all streams and connect them to stdout. "
    "Possible values are 'Trace', 'Debug', 'Info', 'Warning', 'Error' and 'Mute'.";
  icl_core::config::addParameter(icl_core::config::GetoptParameter("log-level:", "l", help_text));
  icl_core::config::addParameter(icl_core::config::GetoptParameter(
                                   "quick-debug:", "qd",
                                   "Activate the QuickDebug log stream and write it "
                                   "to the specified file."));
}

LoggingManager::~LoggingManager()
{
  shutdown();
}

void LoggingManager::printConfiguration() const
{
  std::cerr << "LoggingManager configuration:" << std::endl;

  std::cerr << "  Log output stream factories:" << std::endl;
  for (LogOutputStreamFactoryMap::const_iterator it = m_log_output_stream_factories.begin();
       it != m_log_output_stream_factories.end(); ++it)
  {
    std::cerr << "    " << it->first << std::endl;
  }

  std::cerr << "  Log output streams:" << std::endl;
  if (m_default_log_output)
  {
    m_default_log_output->printConfiguration();
  }
  for (LogOutputStreamMap::const_iterator it = m_log_output_streams.begin();
       it != m_log_output_streams.end(); ++it)
  {
    it->second->printConfiguration();
  }

  std::cerr << "  Log streams:" << std::endl;
  for (LogStreamMap::const_iterator it = m_log_streams.begin(); it != m_log_streams.end(); ++it)
  {
    std::cerr << "    " << it->first << " -> ";
    it->second->printConfiguration();
    std::cerr << std::endl;
  }
}

void LoggingManager::changeLogFormat(const ::icl_core::String& name, const char *format)
{
  for (LogOutputStreamMap::const_iterator it = m_log_output_streams.begin();
       it != m_log_output_streams.end(); ++it)
  {
    if (it->first == name)
    {
      it->second->changeLogFormat(format);
    }
  }
}

void LoggingManager::shutdown()
{
  m_initialized = false;
  m_shutdown_running = true;

  // If the default log output stream exists then remove it from all connected
  // log streams and delete it afterwards.
  if (m_default_log_output != 0)
  {
    removeLogOutputStream(m_default_log_output, false);
    m_default_log_output->shutdown();
    delete m_default_log_output;
    m_default_log_output = 0;
  }

  // Remove all log output streams from all connected log streams and delete
  // the output streams afterwards.
  for (LogOutputStreamMap::iterator output_stream_it = m_log_output_streams.begin();
       output_stream_it != m_log_output_streams.end();
       ++output_stream_it)
  {
    removeLogOutputStream(output_stream_it->second, false);
    output_stream_it->second->shutdown();
    delete output_stream_it->second;
  }

  // Clear the log output stream map.
  m_log_output_streams.clear();

  // Delete all log streams.
  for (LogStreamMap::iterator log_stream_it = m_log_streams.begin();
       log_stream_it != m_log_streams.end();
       ++log_stream_it)
  {
    delete log_stream_it->second;
  }

  // Clear the log stream map.
  m_log_streams.clear();

  m_shutdown_running = false;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  LoggingManager& LoggingManager::Instance()
  {
    return instance();
  }

  /*! Configures log streams and log output streams.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::Configure()
  {
    configure();
  }

  /*! Initializes the logging manager.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::Initialize()
  {
    initialize();
  }

  /*! Check if the logging manager has already been initialized.
   *  \deprecated Obsolete coding style.
   */
  bool LoggingManager::Initialized() const
  {
    return initialized();
  }

  /*! Check if the logging manager has already been initialized.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::AssertInitialized() const
  {
    assertInitialized();
  }

  /*! Registers a log output stream factory with the manager.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::RegisterLogOutputStream(const icl_core::String& name,
                                               LogOutputStreamFactory factory)
  {
    registerLogOutputStream(name, factory);
  }

  /*! Removes a log output stream from the logging manager.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::RemoveLogOutputStream(LogOutputStream *log_output_stream,
                                             bool remove_from_list)
  {
    removeLogOutputStream(log_output_stream, remove_from_list);
  }

  /*! Registers a log stream factory with the manager.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::RegisterLogStream(const icl_core::String& name,
                                         LogStreamFactory factory)
  {
    registerLogStream(name, factory);
  }

  /*! Registers a log stream with the manager.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::RegisterLogStream(LogStream *log_stream)
  {
    registerLogStream(log_stream);
  }

  /*! Removes a log stream from the logging manager.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::RemoveLogStream(const icl_core::String& log_stream_name)
  {
    removeLogStream(log_stream_name);
  }

  /*! Prints the configuration of log streams and log output streams.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::PrintConfiguration() const
  {
    printConfiguration();
  }

  /*! Changes the log output format of the log output streams.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::ChangeLogFormat(const icl_core::String& name,
                                       const char *format)
  {
    changeLogFormat(name, format);
  }

  /*! Shuts down the logging framework.
   *  \deprecated Obsolete coding style.
   */
  void LoggingManager::Shutdown()
  {
    shutdown();
  }

#endif
/////////////////////////////////////////////////

namespace hidden {

  LogOutputStreamRegistrar::LogOutputStreamRegistrar(const ::icl_core::String& name,
                                                     LogOutputStreamFactory factory)
  {
    LoggingManager::instance().registerLogOutputStream(name, factory);
  }

  LogStreamRegistrar::LogStreamRegistrar(const ::icl_core::String& name, LogStreamFactory factory)
  {
    LoggingManager::instance().registerLogStream(name, factory);
  }

}

LifeCycle::LifeCycle(int &argc, char *argv[])
{
  icl_core::config::initialize(argc, argv, icl_core::config::Getopt::eCLC_Cleanup, icl_core::config::Getopt::ePRC_Relaxed);
  LoggingManager::instance().initialize();
}

LifeCycle::~LifeCycle()
{
  LoggingManager::instance().shutdown();
}

}
}
