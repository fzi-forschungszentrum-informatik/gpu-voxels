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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2005-11-12
 *
 * \brief   Contains global LXRT functions
 *
 *
 */
//----------------------------------------------------------------------
#include "icl_core/os_lxrt.h"

#include <cstring>
#include <fstream>

#ifdef _SYSTEM_POSIX_
#include <sys/mman.h>
#endif

#ifdef _SYSTEM_LXRT_
# include <stdlib.h>
# include <rtai_config.h>
# include <rtai_lxrt.h>
# include <rtai_posix.h>
# include <rtai_version.h>
# if defined(_SYSTEM_LXRT_33_) || defined(_SYSTEM_LXRT_35_)
#  include <mca_lxrt_extension.h>
# endif
#endif

using std::memset;
using std::perror;
using std::printf;
using std::strncmp;

//#define LOCAL_PRINTF PRINTF
#define LOCAL_PRINTF(arg)

namespace icl_core {
namespace os {

bool global_lxrt_available = false;
int hard_timer_running = 0;

void lxrtStartup()
{
#ifdef _SYSTEM_LXRT_
  LOCAL_PRINTF("LXRT startup: Check for LXRT ... ");
  checkForLxrt();
  LOCAL_PRINTF("Done\n");

  LOCAL_PRINTF("LXRT startup: Making the task RT ... ");
  makeThisAnLxrtTask();
  LOCAL_PRINTF("Done\n");
#endif
}

void lxrtShutdown()
{
#ifdef _SYSTEM_LXRT_
  LOCAL_PRINTF("LXRT shutdown: Making the task plain Linux ... ");
  makeThisALinuxTask();
  LOCAL_PRINTF("Done\n");
#endif
}

bool checkKernelModule(const char *name)
{
  std::ifstream modules("/proc/modules");
  char line[200];
  while (modules.good())
  {
    memset(line, 0, sizeof(line));
    modules.getline(line, 200);
    if (!strncmp(line, name, strlen(name)))
    {
      return true;
    }
  }
  return false;
}

bool checkForLxrt(void)
{
#ifdef _SYSTEM_LXRT_
# if defined(_SYSTEM_LXRT_33_) || defined(_SYSTEM_LXRT_35_)
  if (!checkKernelModule("mca_lxrt_extension"))
  {
    printf("LXRT: No mca_lxrt_extension module loaded\n LXRT functions not available!\n");
    return false;
  }
# else
  if (!checkKernelModule("rtai_hal"))
  {
    printf("LXRT: No rtai_hal module loaded\n LXRT functions not available!\n");
    return false;
  }
  if (!checkKernelModule("rtai_lxrt"))
  {
    printf("LXRT: No rtai_lxrt module loaded\n LXRT functions not available!\n");
    return false;
  }
  if (!checkKernelModule("rtai_sem"))
  {
    printf("LXRT: No rtai_sem module loaded\n LXRT functions not available!\n");
    return false;
  }
# endif

  // Only print this the first time the function is called!
  if (!global_lxrt_available)
  {
    printf("LXRT: available\n");
  }

  global_lxrt_available = 1;

  return true;
#else
  // no lxrt configured
  return false;
#endif
}

bool isLxrtAvailable()
{
#ifdef _SYSTEM_LXRT_
  return global_lxrt_available;
#else
  return false;
#endif
}

bool isThisLxrtTask()
{
#ifdef _SYSTEM_LXRT_
  return global_lxrt_available && rt_buddy();
#else
  return false;
#endif
}

bool isThisHRT()
{
#ifdef _SYSTEM_LXRT_
  return global_lxrt_available && rt_buddy() && rt_is_hard_real_time(rt_buddy());
#else
  return false;
#endif
}

bool ensureNoHRT()
{
#ifdef _SYSTEM_LXRT_
  if (isThisHRT())
  {
    rt_make_soft_real_time();
    return true;
  }
#endif
  return false;
}

void makeHRT()
{
#ifdef _SYSTEM_LXRT_
  rt_make_hard_real_time();
#endif
}

int makeThisAnLxrtTask()
{
#ifdef _SYSTEM_LXRT_
  if (!global_lxrt_available)
  {
    printf("LXRT not available: MakeThisAnLXRTTask impossible\n");
    return 0;
  }

  //INFOMSG("Making an LXRT thread\n");
  if (isThisLxrtTask())
  {
    printf("LXRT: this is alread an lxrt task\n");
    return 0;
  }

  rt_allow_nonroot_hrt();

  // Lock the process memory.
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
  {
    perror("LXRT: Could not lock the process memory:");
    return 0;
  }

  //printf("LXRT: setting scheduler to SCHED_FIFO\n");
  struct sched_param mysched;
  mysched.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
  if (sched_setscheduler(0, SCHED_FIFO, &mysched) == -1)
  {
    printf("LXRT: ERROR IN SETTING THE SCHEDULER");
    perror("LXRT: errno");
    exit(1);
  }

  if (!rt_task_init(getpid() + pthread_self_rt(), 100, 0, 0))
  {
    printf("LXRT: CANNOT INIT THREAD \n");
    exit(1);
  }

  if ((hard_timer_running = rt_is_hard_timer_running()))
  {
    printf("WARNING: Timer is already on - not activating.");
  }
  else
  {
#if CONFIG_RTAI_VERSION_MINOR < 5
    rt_set_periodic_mode();
    rt_linux_use_fpu(1);
    //rt_task_use_fpu(rt_task, 1);
#else
    rt_set_oneshot_mode();
#endif
    int hp = start_rt_timer(nano2count(500000));
    printf("LXRT: starting TIMER with hp = %i", hp);
  }

  return 1;
#else
  printf("LXRT: Not compiled for LXRT: Cannot switch this task into an lxrt one.\n");
  return 0;
#endif
}


void makeThisALinuxTask()
{
#ifdef _SYSTEM_LXRT_

  if (isThisLxrtTask())
  {
    // DEM("MakeThisALinuxTask soft\n");
    rt_make_soft_real_time();

    // DEM("MakeThisALinuxTask join\n");
#if CONFIG_RTAI_VERSION_MINOR < 5
    RT_TASK *rt_task = rt_buddy();
    if (rt_task)
    {
      rt_task_delete(rt_task);
    }
#else
    pthread_join_rt(pthread_self_rt(), NULL);
#endif

    /* DO NOT STOP THE TIMER!!
     * AFTER STOPPING THE TIMER, the system will not run correclty any more
     *
     * Above all: we would have to ensure that no other threads are running.
     *
     * DEM("MakeThisALinuxTask timer stop\n");
     * // Stopping timer now
     * if ((hard_timer_running = rt_is_hard_timer_running()))
     * {
     *   DEM("Timer still on, stopping rt timer now.");
     *   stop_rt_timer();
     * }
     * else
     * {
     *   DEM("No timer running, STRANGE -- Doing nothing.");
     * }
     */
  }

#else

  printf("LXRT: Not compiled for LXRT: Cannot switch this task from LXRT to linux.\n");

#endif
}

#ifdef _SYSTEM_LXRT_
struct timeval lxrtGetExecTime()
{
  struct timeval time;
# if defined(_SYSTEM_LXRT_33_) || defined(_SYSTEM_LXRT_35_)
  mcalxrt_exectime(&time);
# else
  RTIME exectime[3];
  rt_get_exectime(rt_agent(), exectime);
  struct timespec ts;
  count2timespec(exectime[0], &ts);
  time.tv_sec = ts.tv_sec;
  time.tv_usec = ts.tv_nsec / 1000;
# endif
  return time;
}
#endif

////////////// DEPRECATED VERSIONS //////////////
#ifdef _IC_BUILDER_DEPRECATED_STYLE_

void LxrtStartup() { lxrtStartup(); }

void LxrtShutdown() { lxrtShutdown(); }

bool CheckKernelModule(const char *name) { return checkKernelModule(name); }

bool CheckForLxrt(void) { return checkForLxrt(); }

bool IsLxrtAvailable() { return isLxrtAvailable(); }

bool IsThisLxrtTask() { return isThisLxrtTask(); }

bool IsThisHRT() { return isThisHRT(); }

#ifdef _SYSTEM_LXRT_
struct timeval LxrtGetExecTime() { return lxrtGetExecTime(); }
#endif

bool EnsureNoHRT() { return ensureNoHRT(); }

void MakeHRT() { makeHRT(); }

int MakeThisAnLxrtTask() { return makeThisAnLxrtTask(); }

void MakeThisALinuxTask() { return makeThisALinuxTask(); }

#endif
/////////////////////////////////////////////////

}
}
