# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("ts_icl_core_thread" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  ts_main.cpp
  ts_Mutex.cpp
  ts_PeriodicThread.cpp
  ts_RWLock.cpp
  ts_Semaphore.cpp
  ts_Thread.cpp
  )

IF(Boost_FOUND)
  IF(BUILD_SHARED_LIBS)
    ICMAKER_LOCAL_CPPDEFINES("-DBOOST_TEST_DYN_LINK")    
  ENDIF(BUILD_SHARED_LIBS)
ENDIF(Boost_FOUND)
ICMAKER_EXTERNAL_DEPENDENCIES(
  Boost_UNIT_TEST_FRAMEWORK
  )

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_logging
  icl_core_thread
  )

ICMAKER_BUILD_TEST()
