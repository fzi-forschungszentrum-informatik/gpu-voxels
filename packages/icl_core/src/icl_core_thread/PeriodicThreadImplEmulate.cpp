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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------
#include "PeriodicThreadImplEmulate.h"

namespace icl_core {
namespace thread {

PeriodicThreadImplEmulate::PeriodicThreadImplEmulate(const icl_core::TimeSpan& period)
  : m_period(period)
{
}

PeriodicThreadImplEmulate::~PeriodicThreadImplEmulate()
{
}

bool PeriodicThreadImplEmulate::setPeriod(const icl_core::TimeSpan& period)
{
  m_period = period;
  return true;
}

void PeriodicThreadImplEmulate::waitPeriod()
{
  icl_core::TimeStamp now = ::icl_core::TimeStamp::now();
  icl_core::TimeStamp next_execution_time = m_last_execution_time + m_period;
  icl_core::TimeSpan wait_time_left = next_execution_time - now;

  // If time has run backwards then set the wait time exactly to
  // the period to prevent long waiting times.
  if (wait_time_left > m_period)
  {
    wait_time_left = m_period;
    next_execution_time = now + m_period;
  }

  icl_core::TimeSpan zero_time_span;
  struct timespec wait_time;
  struct timespec remaining_wait_time;
  while (wait_time_left > zero_time_span)
  {
    wait_time = wait_time_left.timespec();
    icl_core::os::nanosleep(&wait_time, &remaining_wait_time);
    wait_time_left = next_execution_time - ::icl_core::TimeStamp::now();
  }

  m_last_execution_time = icl_core::TimeStamp::now();
}


}
}
