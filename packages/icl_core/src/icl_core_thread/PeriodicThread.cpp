// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-04-08
 */
//----------------------------------------------------------------------
#include "Logging.h"
#include "PeriodicThread.h"
#include "PeriodicThreadImpl.h"

#if defined _SYSTEM_QNX_
# include "PeriodicThreadImplQNX.h"
#elif defined _SYSTEM_POSIX_
# if defined _SYSTEM_LXRT_
#  include "PeriodicThreadImplLxrt.h"
# endif
# if defined _SYSTEM_LINUX_
#  include <linux/version.h>
#  if (LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 25))
#   include "PeriodicThreadImplTimerfd.h"
#  endif
# endif
# include "PeriodicThreadImplEmulate.h"
#elif defined _SYSTEM_WIN32_
# include "PeriodicThreadImplEmulate.h"
#else
# error "No implementation specified for System dependent components"
#endif

using icl_core::logging::endl;

namespace icl_core {
namespace thread {

PeriodicThread::PeriodicThread(const icl_core::String& description,
                               const icl_core::TimeSpan& period,
                               ThreadPriority priority)
  : Thread(description, priority)
{
#if defined (_SYSTEM_QNX_)
  LOGGING_DEBUG_CO(Thread, PeriodicThread, threadInfo(),
                   "Creating QNX periodic thread implementation." << endl);
  m_impl = new PeriodicThreadImplQNX(period);

#elif defined (_SYSTEM_POSIX_)
# if defined (_SYSTEM_LXRT_)
  // Only create an LXRT implementation if the LXRT runtime system is
  // really available. Otherwise create an ACE or POSIX
  // implementation, depending on the system configuration.
  // Remark: This allows us to compile programs with LXRT support but
  // run them on systems on which no LXRT is installed and to disable
  // LXRT support at program startup on systems with installed LXRT!
  if (icl_core::os::isLxrtAvailable())
  {
    LOGGING_DEBUG_CO(IclCoreThread, PeriodicThread, threadInfo(),
                     "Creating LXRT periodic thread implementation." << endl);
    m_impl = new PeriodicThreadImplLxrt(period);
  }
  else
  {
# endif
# if defined (_SYSTEM_LINUX_)
#  if (LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 25))
    // Linux system with timerfd in non LXRT mode.
    LOGGING_DEBUG_CO(IclCoreThread, PeriodicThread, threadInfo(),
                     "Creating timerfd periodic thread implementation." << endl);
    m_impl = new PeriodicThreadImplTimerfd(period);
#  else
    // Older Linux system in non LXRT mode.
    LOGGING_DEBUG_CO(IclCoreThread, PeriodicThread, threadInfo(),
                     "Creating emulate periodic thread implementation." << endl);
    m_impl = new PeriodicThreadImplEmulate(period);
#  endif
# else
    // Generic POSIX system.
    LOGGING_DEBUG_CO(IclCoreThread, PeriodicThread, threadInfo(),
                     "Creating emulate periodic thread implementation." << endl);
    m_impl = new PeriodicThreadImplEmulate(period);
# endif
# if defined(_SYSTEM_LXRT_)
  }
# endif

#elif defined (_SYSTEM_WIN32_)
  LOGGING_DEBUG_CO(IclCoreThread, PeriodicThread, threadInfo(),
                   "Creating emulate periodic thread implementation." << endl);
  m_impl = new PeriodicThreadImplEmulate(period);
#endif
}

PeriodicThread::~PeriodicThread()
{
  delete m_impl; m_impl = NULL;
}

icl_core::TimeSpan PeriodicThread::period() const
{
  return m_impl->period();
}

bool PeriodicThread::setPeriod(const icl_core::TimeSpan& period)
{
  return m_impl->setPeriod(period);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Returns the thread's period.
 *  \deprecated Obsolete coding style.
 */
icl_core::TimeSpan PeriodicThread::Period() const
{
  return period();
}

/*! Changes the thread's period.
 *  \deprecated Obsolete coding style.
 */
bool PeriodicThread::SetPeriod(const icl_core::TimeSpan& period)
{
  return setPeriod(period);
}

#endif
/////////////////////////////////////////////////

void PeriodicThread::waitPeriod()
{
  LOGGING_TRACE_CO(IclCoreThread, PeriodicThread, threadInfo(), "Begin." << endl);

  // Reset to hard or soft realtime mode, if necessary.
  if (isHardRealtime() && !executesHardRealtime())
  {
    if (setHardRealtime(true))
    {
      LOGGING_INFO_CO(IclCoreThread, PeriodicThread, threadInfo(),
                      "Resetted to hard realtime mode." << endl);
    }
    else
    {
      LOGGING_ERROR_CO(IclCoreThread, PeriodicThread, threadInfo(),
                       "Resetting to hard realtime mode failed!" << endl);
    }
  }
  else if (!isHardRealtime() && executesHardRealtime())
  {
    if (setHardRealtime(false))
    {
      LOGGING_INFO_CO(IclCoreThread, PeriodicThread, threadInfo(),
                      "Resetted to soft realtime mode." << endl);
    }
    else
    {
      LOGGING_ERROR_CO(IclCoreThread, PeriodicThread, threadInfo(),
                       "Resetting to soft realtime mode failed!" << endl);
    }
  }

  // Wait!
  m_impl->waitPeriod();

  LOGGING_TRACE_CO(IclCoreThread, PeriodicThread, threadInfo(), "Done." << endl);
}

void PeriodicThread::makePeriodic()
{
  LOGGING_TRACE_CO(IclCoreThread, PeriodicThread, threadInfo(), "Begin." << endl);

  m_impl->makePeriodic();

  LOGGING_TRACE_CO(IclCoreThread, PeriodicThread, threadInfo(), "Done." << endl);
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

/*! Sleep until the end of the current period.
 *  \deprecated Obsolete coding style.
 */
void PeriodicThread::WaitPeriod()
{
  waitPeriod();
}

#endif
/////////////////////////////////////////////////

}
}
