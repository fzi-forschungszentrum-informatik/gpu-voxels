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
 * \date    2011-06-06
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_ACTIVE_OBJECT_H_INCLUDED
#define ICL_CORE_THREAD_ACTIVE_OBJECT_H_INCLUDED

#include <icl_core/List.h>

#include "icl_core_thread/ActiveOperation.h"
#include "icl_core_thread/ImportExport.h"
#include "icl_core_thread/Mutex.h"
#include "icl_core_thread/Sem.h"
#include "icl_core_thread/Thread.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

/*! Implements the generic part of the "Active Object" pattern.
 */
class ICL_CORE_THREAD_IMPORT_EXPORT ActiveObject : public Thread
{
public:
  /*! Initialize an active object with a \a description and thread \a
   *  priority.
   */
  ActiveObject(const icl_core::String& description,
               icl_core::ThreadPriority priority = 0);

  /*! Processes the operation queue.  This function should not be
   *  overridden by subclasses.
   */
  virtual void run();

  /*! Subclasses can override this virtual function if they need to do
   *  some processing in the active object thread before the thread
   *  starts to process the operation queue.
   */
  virtual void onThreadStart() { }

  /*! Subclasses can override this virtual function if they need to do
   *  some processing after the active object thread has stopped to
   *  process the operation queue but before the thread stops.
   */
  virtual void onThreadStop() { }

  /*! Stop the active object thread.
   */
  void stop();

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  ICL_CORE_VC_DEPRECATE_STYLE void Stop() ICL_CORE_GCC_DEPRECATE_STYLE
  { stop(); }

#endif
  /////////////////////////////////////////////////

protected:
  /*!
   * Queue a new active operation for future execution.
   */
  void queue(ActiveOperation *active_operation);

  icl_core::List<ActiveOperation*> m_operation_queue;
  Mutex m_operation_queue_mutex;
  Semaphore m_sem;
};

}
}

#endif
