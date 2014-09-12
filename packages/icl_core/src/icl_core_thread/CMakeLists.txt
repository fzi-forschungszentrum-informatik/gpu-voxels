# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_thread" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
    ActiveObject.cpp
    Logging.cpp
    Mutex.cpp
    PeriodicThread.cpp
    PeriodicThreadImplEmulate.cpp
    RWLock.cpp
    ScopedMutexLock.cpp
    ScopedRWLock.cpp
    Sem.cpp
    Thread.cpp
    )

ICMAKER_ADD_HEADERS(
    ActiveObject.h
    ActiveOperation.h
    icl_core_thread.h
    ImportExport.h
    Mutex.h
    PeriodicThread.h
    RWLock.h
    ScopedMutexLock.h
    Sem.h
    SingletonThreadingModels.h
    Thread.h
    )

IF(ICMAKER_DEPRECATED_STYLE)
ICMAKER_ADD_HEADERS(
    tMutex.h
    tPeriodicThread.h
    tRWLock.h
    tScopedMutexLock.h
    tScopedRWLock.h
    tSemaphore.h
    tThread.h
)
ENDIF()

IF(UNIX)
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT Pthread)
ENDIF()

IF(UNIX AND APPLE)
  ICMAKER_ADD_SOURCES(SemaphoreImplDarwin.cpp)
ENDIF(UNIX AND APPLE)

IF(UNIX AND NOT APPLE)
  ICMAKER_ADD_SOURCES(
    PeriodicThreadImplTimerfd.cpp
    SemaphoreImplPosix.cpp
  )
ENDIF(UNIX AND NOT APPLE)

IF(UNIX)
  ICMAKER_ADD_SOURCES(
    MutexImplPosix.cpp
    RWLockImplPosix.cpp
    ThreadImplPosix.cpp )
ENDIF(UNIX)

IF(WIN32)
  ICMAKER_ADD_SOURCES(
    MutexImplWin32.cpp
    PeriodicThreadImplEmulate.cpp
    RWLockImplWin32.cpp
    SemaphoreImplWin32.cpp
    ThreadImplWin32.cpp )
ENDIF(WIN32)

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_THREAD_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_THREAD_)
ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_EXTERNAL_DEPENDENCIES(Threads)

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_logging
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(${icmaker_target})
