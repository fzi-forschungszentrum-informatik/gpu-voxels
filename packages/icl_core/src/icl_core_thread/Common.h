// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 * \brief   Contains icl_core::thread::tMutex
 *
 * \b icl_core::thread::tMutex
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_COMMON_H_INCLUDED
#define ICL_CORE_THREAD_COMMON_H_INCLUDED

#include "icl_core/TimeSpan.h"
#include "icl_core/TimeStamp.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
/*! Operating system independent threading framework with realtime
 *  support.
 */
namespace thread {
//! Namespace for internal implementation details.
namespace impl {

inline TimeSpan getRelativeTimeout(const TimeStamp& timeout_absolute)
{
  TimeStamp now = TimeStamp::now();
  if (timeout_absolute < now)
  {
    return TimeSpan();
  }
  else
  {
    return timeout_absolute - now;
  }
}

inline TimeStamp getAbsoluteTimeout(const TimeSpan& timeout_relative)
{
  TimeStamp timeout_absolute = TimeStamp::now();
  if (timeout_relative < TimeSpan())
  {
    timeout_absolute += TimeSpan(365 * 86400, 0);
  }
  else
  {
    timeout_absolute += timeout_relative;
  }
  return timeout_absolute;
}

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

inline TimeSpan GetRelativeTimeout(const TimeStamp& timeout_absolute) ICL_CORE_GCC_DEPRECATE_STYLE;
ICL_CORE_VC_DEPRECATE_STYLE inline TimeSpan GetRelativeTimeout(const TimeStamp& timeout_absolute)
{ return getRelativeTimeout(timeout_absolute); }

inline TimeStamp GetAbsoluteTimeout(const TimeSpan& timeout_relative) ICL_CORE_GCC_DEPRECATE_STYLE;
ICL_CORE_VC_DEPRECATE_STYLE inline TimeStamp GetAbsoluteTimeout(const TimeSpan& timeout_relative)
{ return getAbsoluteTimeout(timeout_relative); }

#endif
/////////////////////////////////////////////////

}
}
}

#endif
