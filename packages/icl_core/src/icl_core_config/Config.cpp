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
 * \date    2008-11-03
 */
//----------------------------------------------------------------------
#include "icl_core_config/Config.h"

#include <iostream>

namespace icl_core {
namespace config {

const char * CONFIGFILE_CONFIG_KEY = "/configfile";

void dump()
{
  ConfigManager::instance().dump();
}

void debugOutCommandLine(int argc, char *argv[])
{
  for (int j = 0; j < argc; j++)
  {
    std::cout << argv[j] << " ";
  }
  std::cout << std::endl;
}

ConfigIterator find(const ::icl_core::String& query)
{
  return ConfigManager::instance().find(query);
}

bool initialize(int& argc, char *argv[], bool remove_read_arguments)
{
  return initialize(argc, argv,
                    remove_read_arguments ? Getopt::eCLC_Cleanup : Getopt::eCLC_None,
                    Getopt::ePRC_Strict);
}

bool initialize(int& argc, char *argv[], Getopt::CommandLineCleaning cleanup,
                Getopt::ParameterRegistrationCheck registration_check)
{
  // Ensure that the commandline options for ConfigManager are registered.
  ConfigManager::instance();

  bool res = Getopt::instance().initialize(argc, argv, cleanup, registration_check);
  if (res)
  {
    res = ConfigManager::instance().initialize();
  }
  return res;
}

}}
