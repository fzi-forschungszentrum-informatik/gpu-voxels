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
#ifndef ICL_CORE_DISPATCH_PERIODIC_DISPATCH_THREAD_H_INCLUDED
#define ICL_CORE_DISPATCH_PERIODIC_DISPATCH_THREAD_H_INCLUDED

#include "icl_core/List.h"
#include "icl_core_dispatch/ImportExport.h"
#include "icl_core_thread/PeriodicThread.h"

namespace icl_core {
namespace dispatch {

class Operation;

//! Dispatches the contained operations periodically.
class ICL_CORE_DISPATCH_IMPORT_EXPORT PeriodicDispatchThread : public icl_core::thread::PeriodicThread
{
public:
  /*! Initializes a periodic dispatch thread.
   *
   * \param description The thread's description.
   * \param period      The relative period after which the thread is cyclically woken up.
   * \param priority    The thread's priority.
   */
  PeriodicDispatchThread(icl_core::String const & description,
                         icl_core::TimeSpan const & period,
                         ThreadPriority priority = 0);
  //! Destroys a periodic dispatch thread.
  virtual ~PeriodicDispatchThread();

  //! Add the operation \a op to the dispatch queue.
  void addOperation(Operation * op);

private:
  //! Perform the periodic dispatching.
  virtual void run();

  icl_core::List<Operation*> m_dispatch_queue;
};

}
}

#endif
