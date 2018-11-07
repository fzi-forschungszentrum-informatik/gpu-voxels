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
#include "icl_core_thread/ActiveObject.h"

namespace icl_core {
namespace thread {

ActiveObject::ActiveObject(const icl_core::String& description, icl_core::ThreadPriority priority)
  : Thread(description, priority),
    m_sem(0)
{
}

void ActiveObject::run()
{
  onThreadStart();

  // Continually process the operation queue.
  while (execute())
  {
    m_sem.wait();
    if (execute())
    {
      // Process all queued operations.
      while (!m_operation_queue.empty())
      {
        // Try to lock the queue mutex
        if (m_operation_queue_mutex.lock())
        {
          // Extract the next pending operation from the queue.
          ActiveOperation *op = m_operation_queue.front();
          m_operation_queue.pop_front();

          // Release the mutex before executing the operation!
          m_operation_queue_mutex.unlock();

          // Finally, execute the operation.
          op->invoke();
          delete op;
        }
      }
    }
  }

  // Delete any pending operations.
  while (!m_operation_queue.empty())
  {
    delete m_operation_queue.front();
    m_operation_queue.pop_front();
  }

  onThreadStop();
}

void ActiveObject::stop()
{
  Thread::stop();
  m_sem.post();
}

void ActiveObject::queue(ActiveOperation *active_operation)
{
  if (execute() && m_operation_queue_mutex.lock())
  {
    m_operation_queue.push_back(active_operation);
    m_operation_queue_mutex.unlock();
    m_sem.post();
  }
  else
  {
    delete active_operation;
  }
}

}
}
