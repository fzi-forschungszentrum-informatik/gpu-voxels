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
 * \author  Lars Pfotzer <pfotzer@fzi.de>
 * \date    2010-02-08
 */
//----------------------------------------------------------------------

#if defined _SYSTEM_POSIX_
# include <linux/version.h>
# if (!(LINUX_VERSION_CODE < KERNEL_VERSION(2, 6, 25)))

#include "PeriodicThreadImplTimerfd.h"
#include "Logging.h"

#include <sys/timerfd.h>

namespace icl_core {
namespace thread {

PeriodicThreadImplTimerfd::PeriodicThreadImplTimerfd(const icl_core::TimeSpan& period)
  : m_period(period),
    timer_created(false)
{
  m_info = new struct periodic_info;
}

PeriodicThreadImplTimerfd::~PeriodicThreadImplTimerfd()
{
  delete m_info;
}

void PeriodicThreadImplTimerfd::makePeriodic()
{
  /* Create the timer */
  int fd = timerfd_create(CLOCK_MONOTONIC, 0);
  m_info->wakeups_missed = 0;
  m_info->timer_fd = fd;
  if (fd != -1)
  {
    timer_created = true;
  }

  setPeriod(m_period);
}

bool PeriodicThreadImplTimerfd::setPeriod(const icl_core::TimeSpan& period)
{
  m_period = period;

  int ret = -1;
  if (timer_created)
  {
    /* Make the timer periodic */
    unsigned int ns;
    unsigned int sec;
    struct itimerspec itval;

    sec = period.tsSec();
    ns = period.tsNSec();
    itval.it_interval.tv_sec = sec;
    itval.it_interval.tv_nsec = ns;
    itval.it_value.tv_sec = sec;
    itval.it_value.tv_nsec = ns;
    ret = timerfd_settime(m_info->timer_fd, 0, &itval, NULL);
  }
  if (ret == -1)
  {
    return false;
  }
  else
  {
    return true;
  }
}

void PeriodicThreadImplTimerfd::waitPeriod()
{
  unsigned long long missed;
  int ret;

  /* Wait for the next timer event. If we have missed any the
     number is written to "missed" */
  ret = read(m_info->timer_fd, &missed, sizeof(missed));
  if (ret == -1)
  {
    perror ("read timer");
    return;
  }

  m_info->wakeups_missed += missed;
}

}
}

# endif
#endif
