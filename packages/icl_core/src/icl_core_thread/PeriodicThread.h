// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2008-04-08
 *
 * \brief   Contains icl_core::thread::PeriodicThread
 *
 * \b icl_core::thread::PeriodicThread
 *
 * Wrapper class for a periodic thread implementation.
 * Uses system dependent tPeriodicTheradImpl class.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_THREAD_PERIODIC_THREAD_H_INCLUDED
#define ICL_CORE_THREAD_PERIODIC_THREAD_H_INCLUDED

#include <icl_core/BaseTypes.h>
#include <icl_core/Noncopyable.h>
#include <icl_core/TimeSpan.h>
#include <icl_core/os_thread.h>

#include "icl_core_thread/ImportExport.h"
#include "icl_core_thread/Thread.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
namespace thread {

class PeriodicThreadImpl;

/*! Implements a periodic thread.
 *
 *  The thread's main loop can wait until the configured period has
 *  elapsed by calling waitPeriod().
 */
class ICL_CORE_THREAD_IMPORT_EXPORT PeriodicThread : public Thread,
                                                     protected virtual icl_core::Noncopyable
{
public:
  /*! Initializes a periodic thread.
   *
   * \param description The thread's description.
   * \param period The relative period after which the thread is
   *               cyclically woken up.
   * \param priority The thread's priority.
   */
  PeriodicThread(const icl_core::String& description, const icl_core::TimeSpan& period,
                 ThreadPriority priority = 0);

  /*! Deletes a periodic thread.
   */
  virtual ~PeriodicThread();

  /*! Returns the thread's period.
   */
  icl_core::TimeSpan period() const;

  /*! Changes the thread's period.
   *
   *  \return \c true on success, \c false if the period could not be
   *          changed.
   */
  bool setPeriod(const icl_core::TimeSpan& period);

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Returns the thread's period.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE icl_core::TimeSpan Period() const
    ICL_CORE_GCC_DEPRECATE_STYLE;

  /*! Changes the thread's period.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE bool SetPeriod(const icl_core::TimeSpan& period)
    ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

protected:
  /*! Sleep until the end of the current period.
   */
  void waitPeriod();

  ////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

  /*! Sleep until the end of the current period.
   *  \deprecated Obsolete coding style.
   */
  ICL_CORE_VC_DEPRECATE_STYLE void WaitPeriod() ICL_CORE_GCC_DEPRECATE_STYLE;

#endif
  /////////////////////////////////////////////////

private:
  virtual void makePeriodic();

  PeriodicThreadImpl *m_impl;
};

}
}

#endif
