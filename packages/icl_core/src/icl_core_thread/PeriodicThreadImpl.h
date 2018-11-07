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
 * \date    2008-04-15
 *
 * \brief   Contains icl_core::thread::PeriodicThreadImpl
 *
 * \b icl_core::thread::PeriodicThreadImpl
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_PERIODIC_THREAD_IMPL_H_INCLUDED
#define ICL_CORE_THREAD_PERIODIC_THREAD_IMPL_H_INCLUDED

#include <icl_core/Noncopyable.h>
#include <icl_core/TimeSpan.h>

namespace icl_core {
namespace thread {

class PeriodicThreadImpl : protected virtual icl_core::Noncopyable
{
public:
  virtual ~PeriodicThreadImpl() {}

  virtual void makePeriodic() = 0;
  virtual icl_core::TimeSpan period() const = 0;
  virtual bool setPeriod(const icl_core::TimeSpan& period) = 0;
  virtual void waitPeriod() = 0;
};

}
}

#endif
