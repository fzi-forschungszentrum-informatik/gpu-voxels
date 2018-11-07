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
 */
//----------------------------------------------------------------------
#include "PeriodicThreadImplQNX.h"

#include <sys/siginfo.h>
#include <sys/neutrino.h>

#include "Logging.h"

namespace icl_core {
namespace thread {

PeriodicThreadImplQNX::PeriodicThreadImplQNX(const icl_core::TimeSpan& period)
  : m_period(period),
    m_made_periodic(false),
    m_chid(-1)
{
  m_chid = ChannelCreate(0);
  if (m_chid == -1)
  {
    LOGGING_ERROR_C(Thread, PeriodicThreadImplQNX,
                    "Could not create timer channel." << endl);
  }
}

PeriodicThreadImplQNX::~PeriodicThreadImplQNX()
{
  ChannelDestroy(m_chid);
}

void PeriodicThreadImplQNX::makePeriodic()
{
  if (m_chid == -1)
  {
    LOGGING_ERROR_C(Thread, PeriodicThreadImplQNX,
                    "No timer channel available! Cannot make this thread periodic!" << endl);
    return;
  }

  struct sigevent event;
  int coid;

  coid = ConnectAttach(0, 0, m_chid, 0, 0);
  if (coid == -1)
  {
    LOGGING_ERROR_C(Thread, PeriodicThreadImplQNX,
                    "Could not attach to the timer channel! Cannot make this thread periodic!" << endl);
    return;
  }

  SIGEV_PULSE_INIT(&event, coid, SIGEV_PULSE_PRIO_INHERIT, ePT_PULSE_TIMER, 0);

  int res = timer_create(CLOCK_REALTIME, &event, &m_timerid);
  if (res == -1)
  {
    LOGGING_ERROR_C(Thread, PeriodicThreadImplQNX,
                    "Could not create timer! Cannot make this thread periodic!" << endl);
    return;
  }

  m_made_periodic = true;

  setPeriod(m_period);
}

bool PeriodicThreadImplQNX::setPeriod(const icl_core::TimeSpan& period)
{
  m_period = period;

  if (m_made_periodic)
  {
    struct itimerspec timer;
    timer.it_value = m_period.timespec();
    timer.it_interval = m_period.timespec();
    timer_settime(m_timerid, 0, &timer, NULL);
  }

  return true;
}

void PeriodicThreadImplQNX::waitPeriod()
{
  int rcvid;
  struct _pulse msg;

  // TODO: Catch missed periods!
  rcvid = MsgReceivePulse(m_chid, &msg, sizeof(msg), NULL);

  return;
}

}
}
