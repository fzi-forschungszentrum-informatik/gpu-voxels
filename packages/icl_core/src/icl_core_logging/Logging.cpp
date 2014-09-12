// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-11-18
 */
//----------------------------------------------------------------------
#include "icl_core_logging/Logging.h"

#include <icl_core/os_lxrt.h>
#include <icl_core/os_string.h>
#include <icl_core_config/Config.h>

namespace icl_core {
namespace logging {

ThreadStream& operator << (ThreadStream& stream, const icl_core::TimeStamp& time_stamp)
{
#ifdef _SYSTEM_LXRT_
  // Don't use FormatIso8601() in a hard realtime LXRT task, because
  // it might use a POSIX mutex!
  if (icl_core::os::isThisLxrtTask() && icl_core::os::isThisHRT())
  {
    char time_buffer[100];
    memset(time_buffer, 0, 100);
    icl_core::os::snprintf(time_buffer, 99, "%d %02d:%02d:%02d(HRT)",
                           int(time_stamp.days()),
                           int(time_stamp.hours()),
                           int(time_stamp.minutes()),
                           int(time_stamp.seconds()));
    stream << time_buffer;
  }
  else
#endif
  {
    stream << time_stamp.formatIso8601();
  }

  return stream;
}

ThreadStream& operator << (ThreadStream& stream, const icl_core::TimeSpan& time_span)
{
  int64_t calc_secs = time_span.tsSec();
  int64_t calc_nsec = time_span.tsNSec();
  if (calc_secs < 0)
  {
    stream << "-";
    calc_secs = -calc_secs;
  }
  if (calc_secs > 3600)
  {
    stream << calc_secs / 3600 << "h";
    calc_secs = calc_secs % 3600;
  }
  if (calc_secs > 60)
  {
    stream << calc_secs / 60 << "m";
    calc_secs=calc_secs % 60;
  }
  if (calc_secs > 0)
  {
    stream << calc_secs << "s";
  }

  if (calc_nsec / 1000000 * 1000000 == calc_nsec)
  {
    stream << calc_nsec / 1000000 << "ms";
  }
  else if (calc_nsec / 1000 * 1000 == calc_nsec)
  {
    stream << calc_nsec << "us";
  }
  else
  {
    stream << calc_nsec << "ns";
  }

  return stream;
}

REGISTER_LOG_STREAM(Default)
REGISTER_LOG_STREAM(Nirwana)
REGISTER_LOG_STREAM(QuickDebug)

bool initialize(int &argc, char *argv[], bool remove_read_arguments)
{
  return icl_core::logging::initialize(
    argc, argv,
    remove_read_arguments ? icl_core::config::Getopt::eCLC_Cleanup : icl_core::config::Getopt::eCLC_None,
    icl_core::config::Getopt::ePRC_Strict);
}

bool initialize(int &argc, char *argv[],
                icl_core::config::Getopt::CommandLineCleaning cleanup,
                icl_core::config::Getopt::ParameterRegistrationCheck registration_check)
{
  bool result = icl_core::config::initialize(argc, argv, cleanup, registration_check);
  LoggingManager::instance().initialize();
  return result;
}

void initialize()
{
  LoggingManager::instance().initialize();
}

void shutdown()
{
  LoggingManager::instance().shutdown();
}

boost::shared_ptr<LifeCycle> autoStart(int &argc, char *argv[])
{
  return boost::shared_ptr<LifeCycle>(new LifeCycle(argc, argv));
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

bool Initialize(int &argc, char *argv[], bool remove_read_arguments)
{
  return ::icl_core::logging::initialize(argc, argv, remove_read_arguments);
}

bool Initialize(int &argc, char *argv[],
                icl_core::config::Getopt::CommandLineCleaning cleanup,
                icl_core::config::Getopt::ParameterRegistrationCheck registration_check)
{
  return ::icl_core::logging::initialize(argc, argv, cleanup, registration_check);
}

void Initialize() { initialize(); }

void Shutdown() { shutdown(); }

#endif
/////////////////////////////////////////////////

}
}
