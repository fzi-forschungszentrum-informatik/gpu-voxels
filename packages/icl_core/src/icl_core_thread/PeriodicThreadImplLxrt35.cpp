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
 * \date    2010-06-30
 */
//----------------------------------------------------------------------
#include "PeriodicThreadImplLxrt35.h"

#include <rtai_lxrt.h>

#include <icl_core/os_lxrt.h>

namespace icl_core {
namespace thread {

PeriodicThreadImplLxrt35::PeriodicThreadImplLxrt35(const icl_core::TimeSpan& period)
  : m_period(period)
{
}

PeriodicThreadImplLxrt35::~PeriodicThreadImplLxrt35()
{
}

void PeriodicThreadImplLxrt35::makePeriodic()
{
  rt_task_make_periodic_relative_ns(rt_buddy(), 0, m_period.toNSec());
}

bool PeriodicThreadImplLxrt35::setPeriod(const icl_core::TimeSpan& period)
{
  m_period = period;
  return true;
}

void PeriodicThreadImplLxrt35::waitPeriod()
{
  while (rt_task_wait_period())
  { }
}


}
}
