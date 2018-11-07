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
 * \date    2014-12-12
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_SPIN_LOCK_H_INCLUDED
#define ICL_CORE_THREAD_SPIN_LOCK_H_INCLUDED

#include <boost/atomic.hpp>

namespace icl_core {
namespace thread {

/*! A spin lock for very short critical sections.  Busy-waits until
 *  the lock is acquired.  Modeled after the example given in the
 *  boost::atomic documentation.
 */
class SpinLock
{
public:
  SpinLock()
    : m_state(UNLOCKED)
  { }

  /*! Locks the mutex.  Busy-waits until the mutex can be locked.
   *  \returns \c true.  When the function returns the mutex is
   *           guaranteed to be locked.
   */
  bool lock()
  {
    while (m_state.exchange(LOCKED, boost::memory_order_acquire) == LOCKED)
    {
      // busy-wait
    }
    return true;
  }

  /*! Tries to lock the mutex without blocking.  The mutex is locked if
   *  it is available.
   *  \returns \c true if the mutex has been locked, \c false
   *           otherwise.
   */
  bool tryLock()
  {
    return (m_state.exchange(LOCKED, boost::memory_order_acquire) == LOCKED);
  }

  //! Unlocks the mutex.
  void unlock()
  {
    m_state.store(UNLOCKED, boost::memory_order_release);
  }

private:
  //! States of the lock.
  enum LockState
  {
    LOCKED,
    UNLOCKED
  };

  //! Current lock state.
  boost::atomic<LockState> m_state;
};

}
}

#endif
