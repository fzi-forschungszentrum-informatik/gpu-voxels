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
 * \date    2008-11-01
 */
//----------------------------------------------------------------------
#include <iostream>

#include <icl_core_config/Config.h>
#include <icl_core_config/ConfigObserver.h>

class TestObserver : public icl_core::config::ConfigObserver
{
public:
  TestObserver()
  {
    icl_core::config::ConfigManager::instance().registerObserver(this, "/example/config/node");
    icl_core::config::ConfigManager::instance().registerObserver(this, "/observed/by/its/path");
    icl_core::config::ConfigManager::instance().registerObserver(this, "/foo");
    icl_core::config::ConfigManager::instance().registerObserver(this);
  }

  virtual void valueChanged(const icl_core::String &key)
  {
    std::cout << "The value " << key << " changed to ";
    std::cout << icl_core::config::getDefault<icl_core::String>(key, " error when trying to get new value!") << std::endl;
  }
};

int main(int argc, char *argv[])
{
  icl_core::config::ConfigParameterList cmd_parameters;
  cmd_parameters.push_back(icl_core::config::ConfigParameter("example:", "e", "/example/config/node", "Example help text."));
  icl_core::config::addParameter(cmd_parameters);

  icl_core::config::initialize(argc, argv);

  icl_core::config::ConfigManager::instance().dump();

  std::cout << std::endl << "-- Direct access by name --" << std::endl;
  icl_core::String value;
  bool success = icl_core::config::get<icl_core::String>("/example/config/node", value);
  std::cout << "/example/config/node(" << success << "): " << value << std::endl;

  std::cout << std::endl << "-- Direct access by name with default value --" << std::endl;
  icl_core::String another_value = icl_core::config::getDefault<icl_core::String>("/example/config/node", icl_core::String("42"));
  std::cout << "/example/config/node: " << another_value << std::endl;

  std::cout << std::endl << "-- Regex access --" << std::endl;
  icl_core::config::ConfigIterator find_it = icl_core::config::find("\\/example\\/(.*)");
  while (find_it.next()) {
    std::cout << find_it.key() << "(" << find_it.matchGroup(1) << "): " << find_it.value() << std::endl;
  }

  std::cout << std::endl << "-- Observing configuration changes --" << std::endl;
  TestObserver observer;
  icl_core::config::ConfigManager::instance().setValue<double>("/observed/by/its/path", 4.2);
  icl_core::config::ConfigManager::instance().setValue<icl_core::String>("/observed/generally", "Using an empty key during registration");

  return 0;
}
