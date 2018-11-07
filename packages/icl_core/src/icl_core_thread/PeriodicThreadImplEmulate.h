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
 * \date    2008-04-08
 *
 * \brief   Contains icl_core::thread::PeriodicThread
 *
 * \b icl_core::thread::PeriodicThread
 *
 * Wrapper class for a periodic thread implementation.
 * Uses system dependent tPeriodicTheradImpl class.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_PERIODIC_THREAD_IMPL_EMULATE_H_INCLUDED
#define ICL_CORE_THREAD_PERIODIC_THREAD_IMPL_EMULATE_H_INCLUDED

#include <icl_core/TimeSpan.h>
#include <icl_core/TimeStamp.h>

#include "icl_core_thread/PeriodicThreadImpl.h"

namespace icl_core {
namespace thread {

class PeriodicThreadImplEmulate : public PeriodicThreadImpl, protected virtual icl_core::Noncopyable
{
public:
  PeriodicThreadImplEmulate(const icl_core::TimeSpan& period);
  virtual ~PeriodicThreadImplEmulate();

  virtual void makePeriodic() { }
  virtual icl_core::TimeSpan period() const { return m_period; }
  virtual bool setPeriod(const icl_core::TimeSpan& period);
  virtual void waitPeriod();

private:
  icl_core::TimeSpan m_period;
  icl_core::TimeStamp m_last_execution_time;
};

}
}

#endif
