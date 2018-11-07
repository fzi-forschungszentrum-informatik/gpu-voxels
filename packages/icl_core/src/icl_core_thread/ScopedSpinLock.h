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
#ifndef ICL_CORE_THREAD_SCOPED_SPIN_LOCK_H_INCLUDED
#define ICL_CORE_THREAD_SCOPED_SPIN_LOCK_H_INCLUDED

#include <icl_core_thread/SpinLock.h>

namespace icl_core {
namespace thread {

class ScopedSpinLock : protected virtual icl_core::Noncopyable
{
public:
  /*! Locks the \a spinlock.
   *  \param spinlock The spinlock to use.
   */
  explicit ScopedSpinLock(SpinLock& spinlock)
    : m_spinlock(spinlock)
  {
    m_spinlock.lock();
  }

  //! Unlocks the spinlock.
  ~ScopedSpinLock()
  {
    m_spinlock.unlock();
  }

  /*! Check if the spinlock has been successfully locked (always true
   *  since this class guarantees that the lock is held).
   */
  bool isLocked() const { return true; }

  //! Implicit conversion to bool (always true).
  operator bool () const { return isLocked(); }

private:
  SpinLock& m_spinlock;
};

}
}

#endif
