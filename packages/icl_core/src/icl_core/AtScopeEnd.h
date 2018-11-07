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
 * \author  Jan Oberlaender <oberlaender@fzi.de>
 * \date    2014-11-03
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_AT_SCOPE_END_H_INCLUDED
#define ICL_CORE_AT_SCOPE_END_H_INCLUDED

#include <boost/function.hpp>

namespace icl_core {

/*! A simple helper class which executes a specific function when the
 *  object runs out of scope.  Sometimes it is much too complex to
 *  wrap some structure that is outside your control into a proper C++
 *  class with a destructor, just to make sure that you never forget
 *  to clean up.  In this case, just create an AtScopeEnd object and
 *  pass to it a matching cleanup function (any thing that can be
 *  called as a void() function).
 */
class AtScopeEnd
{
public:
  //! Create an object which runs \a func when it is destroyed.
  AtScopeEnd(const boost::function<void()>& func)
    : m_func(func),
      m_run_at_scope_end(true)
  { }

  //! Destructor, calls the function as promised.
  ~AtScopeEnd()
  {
    if (m_run_at_scope_end)
    {
      m_func();
    }
  }

  //! Disable the object, do not run the function upon destruction.
  void disable() { m_run_at_scope_end = false; }
  //! Enable the object to run the function upon destruction (the default).
  void enable() { m_run_at_scope_end = true; }

private:
  //! The function to run upon destruction.
  boost::function<void()> m_func;
  //! If \c false, #m_func is \e not run upon destruction.
  bool m_run_at_scope_end;
};

}

#endif
