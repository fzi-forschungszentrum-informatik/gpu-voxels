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
#ifndef ICL_CORE_DISPATCH_OPERATION_QUEUE_H_INCLUDED
#define ICL_CORE_DISPATCH_OPERATION_QUEUE_H_INCLUDED

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include "icl_core/List.h"

namespace icl_core {
namespace thread {

class Operation;
typedef boost::shared_ptr<Operation> OperationPtr;

/** An operation queue contains a set operations that
 *  may, partly, be executed in parallel.
 */
class OperationQueue : boost::noncopyable
{
public:
  /** Create a new operation queue.
   *  \a retain specifies if operations will be deleted from or
   *  retained in the queue after they have been executed.
   */
  OperationQueue(bool retain = false);

  //! Add the operation \a op to the queue.
  void addOperation(OperationPtr op);

private:
  icl_core::List<OperationPtr> m_operations;
};

}
}

#endif

