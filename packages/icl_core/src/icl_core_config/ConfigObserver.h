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
 * \author  Dennis Nienhüser <nienhues@fzi.de>
 * \date    2009-07-08
 *
 * \brief   Contains ConfigObserver.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_OBSERVER_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_OBSERVER_H_INCLUDED

#include "icl_core_config/ConfigManager.h"

namespace icl_core {
namespace config {

//! Interface for observing configuration changes
/*!
 * Implement this interface and register with the ConfigManager
 * singleton instance for one or more config keys. If the
 * value of one of these keys changes, the valueChanged
 * method will be called, which needs to be implemented.
 */
class ConfigObserver
{
public:
  /*!
   * The value of the given configuration key has changed. This method
   * will be called for any key if you registered for it using ConfigManager's
   * RegisterObserver() method
   */
  virtual void valueChanged(const icl_core::String &key) = 0;

  /*!
   * Destructor. Automatically unregisters from ConfigManager
   */
  virtual ~ConfigObserver()
  {
    icl_core::config::ConfigManager::instance().unregisterObserver(this);
  }
};

} // namespace config
} // namespace icl_core

#endif // T_CONFIG_OBSERVER_H
