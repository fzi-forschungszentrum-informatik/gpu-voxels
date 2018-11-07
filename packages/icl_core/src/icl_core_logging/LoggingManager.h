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
 * \date    2007-10-02
 *
 * \brief   Contains icl_logging::LoggingManager
 *
 * \b icl_logging::LoggingManager
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_MANAGER_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MANAGER_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/List.h>
#include <icl_core/Map.h>
#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogLevel.h"

#include <boost/shared_ptr.hpp>

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace logging {

class LogOutputStream;
class LogStream;

typedef LogOutputStream* (*LogOutputStreamFactory)(const icl_core::String& name,
                                                   const icl_core::String& config_prefix,
                                                   LogLevel log_level);
typedef LogStream* (*LogStreamFactory)();

/*! \brief Manages the logging framework.
 *
 *  The logging framework can be configured through a call to
 *  Initialize(). It will then read its configuration via
 *  icl_core_config. See
 *  http://www.mca2.org/wiki/index.php/ProgMan:Logging for the
 *  details.
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT LoggingManager
{
public:
  static LoggingManager& instance()
  {
    static LoggingManager manager_instance;
    return manager_instance;
  }

  /*! Configures log streams and log output streams.
   *
   *  This function is only useful if log streams are created
   *  dynamically after the logging manager has been initialized.
   */
  void configure();

  /*! Initializes the logging manager.
   *
   *  Remark: It is preferred to use the convenience functions
   *  ::icl_core::logging::initialize(),
   *  ::icl_core::logging::initialize(int&, char *[], bool) or
   *  ::icl_core::logging::initialize(int&, char *[],
   *  ::icl_core::config::Getopt::CommandLineCleaning,
   *  ::icl_core::config::Getopt::ParameterRegistrationCheck) instead
   *  of directly calling this method.
   */
  void initialize();

  /*! Check if the logging manager has already been initialized.
   */
  bool initialized() const { return m_initialized; }

  /*! Check if the logging manager has already been initialized.
   *  Aborts the program if not.
   */
  void assertInitialized() const;

  /*! Registers a log output stream factory with the manager.
   */
  void registerLogOutputStream(const icl_core::String& name, LogOutputStreamFactory factory);

  /*! Removes a log output stream from the logging manager.
   */
  void removeLogOutputStream(LogOutputStream *log_output_stream, bool remove_from_list = true);

  /*! Registers a log stream factory with the manager.
   */
  void registerLogStream(const icl_core::String& name, LogStreamFactory factory);

  /*! Registers a log stream with the manager.
   */
  void registerLogStream(LogStream *log_stream);

  /*! Removes a log stream from the logging manager.
   */
  void removeLogStream(const icl_core::String& log_stream_name);

  /*! Prints the configuration of log streams and log output streams.
   *
   *  Remark: This is mainly for debugging purposes!
   */
  void printConfiguration() const;

  /*! Changes the log output format of the log output streams. See
   *  LogOutputStream#changeLogFormat for format definition
   */
  void changeLogFormat(const icl_core::String& name, const char *format = "~T ~S(~L)~ C~(O~::D: ~E");

  /*! Shuts down the logging framework. Any log messages that are
   *  pending in log output streams are written out. The log output
   *  stream threads are then stopped so that no further log messages
   *  are processed.
   */
  void shutdown();

  //! Set the log level globally for all existing streams.
  void setLogLevel(icl_core::logging::LogLevel log_level);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  static ICL_CORE_VC_DEPRECATE_STYLE LoggingManager& Instance()
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Configures log streams and log output streams.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Configure()
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Initializes the logging manager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Initialize()
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check if the logging manager has already been initialized.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool Initialized() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Check if the logging manager has already been initialized.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void AssertInitialized() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Registers a log output stream factory with the manager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RegisterLogOutputStream(const icl_core::String& name,
                                                           LogOutputStreamFactory factory)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Removes a log output stream from the logging manager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RemoveLogOutputStream(LogOutputStream *log_output_stream,
                                                         bool remove_from_list = true)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Registers a log stream factory with the manager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RegisterLogStream(const icl_core::String& name,
                                                     LogStreamFactory factory)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Registers a log stream with the manager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RegisterLogStream(LogStream *log_stream)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Removes a log stream from the logging manager.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void RemoveLogStream(const icl_core::String& log_stream_name)
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Prints the configuration of log streams and log output streams.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void PrintConfiguration() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Changes the log output format of the log output streams.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void ChangeLogFormat(const icl_core::String& name,
                                                   const char *format = "~T ~S(~L)~ C~(O~::D: ~E")
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Shuts down the logging framework.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void Shutdown()
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  typedef icl_core::List<icl_core::String> StringList;

  //! Configuration of a LogOutputStream.
  struct LogOutputStreamConfig
  {
    /*! The name of the output stream class as registered by the
     *  implementation.
     */
    icl_core::String output_stream_name;
    //! The name of the output stream instance which will be created.
    icl_core::String name;
    //! The log level of this output stream.
    LogLevel log_level;
    //! All associated log streams.
    StringList log_streams;
  };
  typedef icl_core::Map<icl_core::String, LogOutputStreamConfig> LogOutputStreamConfigMap;

  //! Configuration of a LogStream.
  struct LogStreamConfig
  {
    //! Name of the log stream.
    icl_core::String name;
    //! Log level of the log stream.
    LogLevel log_level;
  };
  typedef icl_core::Map<icl_core::String, LogStreamConfig> LogStreamConfigMap;

  LoggingManager();

  ~LoggingManager();

  // Forbid copying logging manager objects.
  LoggingManager(const LoggingManager&);
  LoggingManager& operator = (const LoggingManager&);

  bool m_initialized;
  bool m_shutdown_running;

  LogOutputStreamConfigMap m_output_stream_config;
  LogStreamConfigMap m_log_stream_config;

  typedef icl_core::Map<icl_core::String, LogStream*> LogStreamMap;
  typedef icl_core::Map<icl_core::String, LogOutputStreamFactory> LogOutputStreamFactoryMap;
  typedef icl_core::Map<icl_core::String, LogStreamFactory> LogStreamFactoryMap;
  typedef icl_core::Map<icl_core::String, LogOutputStream*> LogOutputStreamMap;
  LogStreamMap m_log_streams;
  LogOutputStreamFactoryMap m_log_output_stream_factories;
  LogStreamFactoryMap m_log_stream_factories;
  LogOutputStreamMap m_log_output_streams;

  LogOutputStream *m_default_log_output;
};

//! Internal namespace for implementation details.
namespace hidden {

/*! Helper class to register a log output stream with the logging
 *  manager.
 *
 *  Remark: Never use this class directly! Use the
 *  REGISTER_LOG_OUTPUT_STREAM() macro instead!
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT LogOutputStreamRegistrar
{
public:
  LogOutputStreamRegistrar(const icl_core::String& name, LogOutputStreamFactory factory);
};

/*! Helper class to register a log stream with the logging manager.
 *
 *  Remark: Never use this class directly! Use the
 *  REGISTER_LOG_STREAM() macro instead!
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT LogStreamRegistrar
{
public:
  LogStreamRegistrar(const icl_core::String& name, LogStreamFactory factory);
};

}

/**
 * Convenience class to manage the initialize() shutdown() sequence
 */
class ICL_CORE_LOGGING_IMPORT_EXPORT LifeCycle
{
public:
  //! Convenience shared pointer shorthand.
  typedef boost::shared_ptr<LifeCycle> Ptr;
  //! Convenience shared pointer shorthand (const version).
  typedef boost::shared_ptr<const LifeCycle> ConstPtr;

  /** Initializes logging and removes known parameters from argc, argv */
  LifeCycle(int &argc, char *argv[]);

  /** Shuts down logging (!) */
  ~LifeCycle();
};

}
}

#endif
