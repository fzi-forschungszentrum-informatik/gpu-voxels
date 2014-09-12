// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-16
 *
 */
//----------------------------------------------------------------------
#include "icl_core_logging/UdpLogOutput.h"

#include <netdb.h>
#include <boost/regex.hpp>
#include <sys/types.h>
#include <sys/socket.h>

#include "icl_core/StringHelper.h"
#include "icl_core_config/Config.h"
#include "icl_core_logging/Logging.h"

namespace icl_core {
namespace logging {

REGISTER_LOG_OUTPUT_STREAM(UDP, &UdpLogOutput::create)

LogOutputStream *UdpLogOutput::create(const icl_core::String& name, const icl_core::String& config_prefix,
                                      icl_core::logging::LogLevel log_level)
{
  return new UdpLogOutput(name, config_prefix, log_level);
}

UdpLogOutput::UdpLogOutput(const icl_core::String& name, const icl_core::String& config_prefix,
                           icl_core::logging::LogLevel log_level)
  : LogOutputStream(name, config_prefix, log_level),
    m_socket(-1)
{
  // Get the server configuration.
  icl_core::String server_host;
  if (!icl_core::config::get<icl_core::String>(config_prefix + "/Host", server_host))
  {
    std::cerr << "No Host specified for UDP log output stream " << config_prefix << std::endl;
  }

  icl_core::String server_port =
    icl_core::config::getDefault<icl_core::String>(config_prefix + "/Port", "60000");

  if (!icl_core::config::get<icl_core::String>(config_prefix + "/SystemName", m_system_name))
  {
    std::cerr << "No SystemName specified for UDP log output stream " << config_prefix << std::endl;
  }

  // Open the UDP socket.
  struct addrinfo hints;
  memset (&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;

  struct addrinfo *res = 0, *res0 = 0;
  int n = getaddrinfo(server_host.c_str (), server_port.c_str (), &hints, &res0);
  if (n == 0)
  {
    for (res = res0; res != NULL && m_socket < 0; res = res->ai_next)
    {
      m_socket = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
      if (m_socket >= 0)
      {
        if (connect(m_socket, res->ai_addr, res->ai_addrlen) < 0)
        {
          close(m_socket);
          m_socket = -1;
        }
      }
    }

    freeaddrinfo(res0);
  }
}

UdpLogOutput::~UdpLogOutput()
{
  if (m_socket >= 0)
  {
    close(m_socket);
  }
}

void UdpLogOutput::pushImpl(const LogMessage& log_message)
{
  if (m_socket >= 0)
  {
    std::stringstream ss;
    ss << "'" << m_system_name << "',"
       << "'" << log_message.timestamp.formatIso8601() << "'," << log_message.timestamp.tsNSec() << ","
       << "'" << logLevelDescription(log_message.log_level) << "',"
       << "'" << log_message.log_stream << "',"
       << "'" << log_message.filename << "'," << log_message.line << ","
       << "'" << log_message.class_name << "',"
       << "'" << escape(log_message.object_name) << "',"
       << "'" << log_message.function_name << "',"
       << "'" << escape(log_message.message_text) << "'";
    std::string str = ss.str();
    int res = write(m_socket, str.c_str(), str.length());
    if (res < 0)
    {
      perror("UdpLogOutput::pushImpl()");
    }
  }
}

icl_core::String UdpLogOutput::escape(icl_core::String str) const
{
  // TODO: Which characters have to be escaped to ensure that a
  // correct SQL statement is created?
  str = boost::regex_replace(str, boost::regex("'"), "\\'");
  //str = boost::regex_replace(str, boost::regex("\\n"), "\\\\n");
  return str;
}

}
}
