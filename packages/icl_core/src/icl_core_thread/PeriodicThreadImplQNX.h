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
 * \date    2010-01-02
 *
 * \brief   Contains icl_core::thread::PeriodicThreadImplQNX
 *
 * \b icl_core::thread::PeriodicThreadImplQNX
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_PERIODIC_THREAD_IMPL_QNX_H_INCLUDED
#define ICL_CORE_THREAD_PERIODIC_THREAD_IMPL_QNX_H_INCLUDED

#include <time.h>

#include "PeriodicThreadImpl.h"

namespace icl_core {
namespace thread {

class PeriodicThreadImplQNX : public PeriodicThreadImpl, protected virtual icl_core::Noncopyable
{
public:
  explicit PeriodicThreadImplQNX(const icl_core::TimeSpan& period);
  virtual ~PeriodicThreadImplQNX();

  virtual void makePeriodic();
  virtual icl_core::TimeSpan period() const { return m_period; }
  virtual bool setPeriod(const icl_core::TimeSpan& period);
  virtual void waitPeriod();

private:
  enum PulseType
  {
    ePT_PULSE_TIMER,
    ePT_PULSE_SELF_SYNC
  };

  icl_core::TimeSpan m_period;
  bool m_made_periodic;

  int m_chid;
  timer_t m_timerid;
};

}
}

#endif
