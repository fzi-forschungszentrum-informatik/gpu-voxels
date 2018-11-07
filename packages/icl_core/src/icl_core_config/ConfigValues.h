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
 * \date    2010-04-28
 *
 * \brief   Contains tConfigValue
 *
 * "Container" class for batch reading of configuration parameters.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_VALUES_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_VALUES_H_INCLUDED

#include <list>
#include <boost/assign/list_of.hpp>

#include "icl_core_config/ConfigEnum.h"
#include "icl_core_config/ConfigEnumDefault.h"
#include "icl_core_config/ConfigList.h"
#include "icl_core_config/ConfigValue.h"
#include "icl_core_config/ConfigValueDefault.h"
#include "icl_core_config/ConfigValueIface.h"

#define CONFIG_VALUES(list) boost::assign::list_of<icl_core::config::impl::ConfigValueIface*> list . operator icl_core::config::ConfigValueList ()

namespace icl_core {
namespace config {

typedef impl::ConfigValueIface* ConfigValues[];
typedef std::list<impl::ConfigValueIface*> ConfigValueList;

}}

#endif
