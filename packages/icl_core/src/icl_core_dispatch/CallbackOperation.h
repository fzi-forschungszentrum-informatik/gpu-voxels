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
 * \date    2012-02-15
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_DISPATCH_CALLBACK_OPERATION_H_INCLUDED
#define ICL_CORE_DISPATCH_CALLBACK_OPERATION_H_INCLUDED

#include <boost/function.hpp>

#include "icl_core_dispatch/Operation.h"

namespace icl_core {
namespace dispatch {

//! Implements an operation that calls back a registered function.
class CallbackOperation : public Operation
{
public:
  //! Create a callback operation using the \a callback function object.
  CallbackOperation(boost::function<void ()> const & callback);
  //! Destroy a callback operation.
  virtual ~CallbackOperation();

  //! Execute the callback function.
  virtual void execute();

private:
  boost::function<void ()> m_callback;
};

}}

#endif
