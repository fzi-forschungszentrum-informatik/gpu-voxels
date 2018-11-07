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
 * \author  Jan Oberländer <oberlaen@fzi.de>
 * \date    2009-06-16
 *
 * \brief   Contains lifetime policies for icl_core::Singleton.
 *
 * \b icl_core::SLPDefaultLifetime schedules the singleton destruction
 * using atexit().
 * \b icl_core::SLPNoDestroy never destroys the singleton.  Unless a
 * static creation policy is used, this means the object is leaked at
 * program termination.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SINGLETON_LIFETIME_POLICIES_H_INCLUDED
#define ICL_CORE_SINGLETON_LIFETIME_POLICIES_H_INCLUDED

#include <stdlib.h>
#include <stdexcept>

namespace icl_core {

//! Helper definition for destruction functions.
typedef void (*DestructionFuncPtr)();

/*! Default lifetime policy for singletons.  The destruction of the
 *  singleton instance is scheduled using atexit().  In case an
 *  instance is accessed after its destruction, std::logic_error is
 *  thrown.
 */
template
<class T>
class SLPDefaultLifetime
{
public:
  static void scheduleDestruction(DestructionFuncPtr f)
  {
    atexit(f);
  }

  static void onDeadReference()
  {
    throw std::logic_error("attempted to access a singleton instance after its destruction");
  }
};

/*! Non-destruction lifetime policy for singletons.  The singleton
 *  instance is never destroyed, unless a static creation policy is
 *  used, in which case The compiler's runtime code will automatically
 *  destroy it.
 */
template
<class T>
class SLPNoDestroy
{
public:
  static void scheduleDestruction(DestructionFuncPtr)
  {
  }

  static void onDeadReference()
  {
  }
};

}

#endif
