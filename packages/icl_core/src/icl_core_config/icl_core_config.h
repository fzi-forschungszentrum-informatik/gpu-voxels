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
 *
 * \brief   Collects all exported header files for use with precompiled headers.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_ICL_CORE_CONFIG_H_INCLUDED
#define ICL_CORE_CONFIG_ICL_CORE_CONFIG_H_INCLUDED

#ifndef _IC_BUILDER_ICL_CORE_CONFIG_
#  define _IC_BUILDER_ICL_CORE_CONFIG_
#endif

#include <boost/regex.h>
#include <tinyxml.h>
#include <icl_core/icl_core.h>

#include "icl_core_config/Config.h"
#include "icl_core_config/ConfigIterator.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigObserver.h"
#include "icl_core_config/ConfigParameter.h"
#include "icl_core_config/ConfigValues.h"
#include "icl_core_config/GetoptParser.h"
#include "icl_core_config/GetoptParameter.h"
#include "icl_core_config/tConfig.h"
#include "icl_core_config/tConfigIterator.h"
#include "icl_core_config/tConfigObserver.h"
#include "icl_core_config/tConfigParameter.h"
#include "icl_core_config/tConfigValues.h"
#include "icl_core_config/tGetopt.h"
#include "icl_core_config/tGetoptParameter.h"

#endif
