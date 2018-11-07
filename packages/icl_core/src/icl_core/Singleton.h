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
 * \brief   Contains icl_core::Singleton
 *
 * \b icl_core::Singleton provides a generic, policy-based singleton.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_SINGLETON_H_INCLUDED
#define ICL_CORE_SINGLETON_H_INCLUDED

#include "icl_core/SingletonCreationPolicies.h"
#include "icl_core/SingletonLifetimePolicies.h"
#include "icl_core/SingletonThreadingModels.h"

namespace icl_core {

/*! A generic, policy-based singleton.
 *  \param T The class for which to maintain a singleton instance.
 *  \param TCreationPolicy A policy for creating the instance, to
 *         allow for different allocators.
 *         \see SingletonCreationPolicies.h
 *  \param TLifetimePolicy A policy for managing the instance's lifetime,
 *         to control whether, when and how to destroy the instance.
 *         \see SingletonLifetimePolicies.h
 *  \param TThreadingModel A policy for managing thread safety.  The standard
 *         model provided in icl_core is for single-threaded use only.
 *         Thread-safe implementations can be found in icl_core_logging and
 *         icl_core_thread.
 *         \see SingletonThreadingModels.h
 */
template
<class T,
 template <class> class TCreationPolicy = SCPCreateUsingNew,
 template <class> class TLifetimePolicy = SLPDefaultLifetime,
 template <class> class TThreadingModel = STMSingleThreaded>
class Singleton
{
public:
  //! Provide access to the singleton instance.
  static T& instance();

private:
  //! Forbid creation.
  Singleton();
  //! Forbid deletion.
  ~Singleton();
  //! Forbid copy-construction.
  Singleton(const Singleton&);
  //! Forbid assignment.
  Singleton& operator = (const Singleton&);

  //! Helper for destroying the instance.
  static void destroySingleton();
  //! The singleton instance.
  static T *m_instance;
  //! Indicates whether the instance has been destroyed.
  static bool m_destroyed;
  //! The lock for synchronizing instantiation.  Must be default-constructable.
  static typename TThreadingModel<T>::Lock m_lock;
};

}

#endif
