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
 * \date    2011-12-12
 *
 */
//----------------------------------------------------------------------
#include "icl_core_dispatch/PeriodicDispatchThread.h"

#include "icl_core_dispatch/Operation.h"

#include <boost/foreach.hpp>

namespace icl_core {
namespace dispatch {

PeriodicDispatchThread::PeriodicDispatchThread(icl_core::String const & description,
                                               icl_core::TimeSpan const & period,
                                               ThreadPriority priority)
  : icl_core::thread::PeriodicThread(description, period, priority)
{
}

PeriodicDispatchThread::~PeriodicDispatchThread()
{
  stop();
  join();
}

void PeriodicDispatchThread::addOperation(Operation *op)
{
  m_dispatch_queue.push_back(op);
}

void PeriodicDispatchThread::run()
{
  while (execute())
  {
    BOOST_FOREACH (Operation *op, m_dispatch_queue)
    {
      op->execute();
    }

    if (execute())
    {
      waitPeriod();
    }
  }
}

}
}
