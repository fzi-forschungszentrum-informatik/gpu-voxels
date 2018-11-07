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
#ifndef ICL_CORE_SINGLETON_HPP_INCLUDED
#define ICL_CORE_SINGLETON_HPP_INCLUDED

#include "Singleton.h"

namespace icl_core {

template
<class T, template <class> class TCreationPolicy,
 template <class> class TLifetimePolicy, template <class> class TThreadingModel>
T& Singleton<T, TCreationPolicy, TLifetimePolicy, TThreadingModel>::instance()
{
  T *temp = m_instance;
  TThreadingModel<T>::memoryBarrier();
  if (temp == NULL)
  {
    typename TThreadingModel<T>::Guard guard(m_lock);
    temp = m_instance;
    if (temp == NULL)
    {
      if (m_destroyed)
      {
        TLifetimePolicy<T>::onDeadReference();
        m_destroyed = false;
      }
      temp = TCreationPolicy<T>::create();
      TThreadingModel<T>::memoryBarrier();
      m_instance = temp;
      TLifetimePolicy<T>::scheduleDestruction(&destroySingleton);
    }
  }
  return *m_instance;
}

template
<class T, template <class> class TCreationPolicy,
 template <class> class TLifetimePolicy, template <class> class TThreadingModel>
void Singleton<T, TCreationPolicy, TLifetimePolicy, TThreadingModel>::destroySingleton()
{
  TCreationPolicy<T>::destroy(m_instance);
  m_instance = NULL;
  m_destroyed = true;
}

template
<class T, template <class> class TCreationPolicy,
 template <class> class TLifetimePolicy, template <class> class TThreadingModel>
T *Singleton<T, TCreationPolicy, TLifetimePolicy, TThreadingModel>::m_instance = NULL;

template
<class T, template <class> class TCreationPolicy,
 template <class> class TLifetimePolicy, template <class> class TThreadingModel>
bool Singleton<T, TCreationPolicy, TLifetimePolicy, TThreadingModel>::m_destroyed = false;

template
<class T, template <class> class TCreationPolicy,
 template <class> class TLifetimePolicy, template <class> class TThreadingModel>
typename TThreadingModel<T>::Lock Singleton<T, TCreationPolicy, TLifetimePolicy, TThreadingModel>::m_lock;

}

#endif
