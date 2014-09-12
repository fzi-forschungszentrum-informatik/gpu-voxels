# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_logging" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  FileLogOutput.cpp
  Logging.cpp
  LoggingManager.cpp
  LogLevel.cpp
  LogOutputStream.cpp
  LogStream.cpp
  ScopedSemaphore.cpp
  Semaphore.cpp
  StdErrorLogOutput.cpp
  StdLogOutput.cpp
  Thread.cpp
  ThreadStream.cpp
  )

ICMAKER_ADD_HEADERS(
  icl_core_logging.h
  ImportExport.h
  Constants.h
  FileLogOutput.h
  Logging.h
  LoggingMacros_LLOGGING.h
  LoggingMacros_LLOGGING_FMT.h
  LoggingMacros_LOGGING.h
  LoggingMacros_LOGGING_FMT.h
  LoggingMacros_MLOGGING.h
  LoggingMacros_MLOGGING_FMT.h
  LoggingMacros_SLOGGING.h
  LoggingMacros_SLOGGING_FMT.h
  LoggingManager.h
  LogLevel.h
  LogOutputStream.h
  LogStream.h
  ScopedSemaphore.h
  ScopedTimer.h
  Semaphore.h
  SingletonThreadingModels.h
  StdErrorLogOutput.h
  StdLogOutput.h
  Thread.h
  ThreadStream.h
  )

IF(ICMAKER_DEPRECATED_STYLE)
ICMAKER_ADD_HEADERS(
  tFileLogOutput.h
  tLoggingManager.h
  tLogLevel.h
  tLogOutputStream.h
  tLogStream.h
  tSemaphore.h
  tScopedSemaphore.h
  tStdErrorLogOutput.h
  tStdLogOutput.h
  tThreadStream.h
  tFileLogOutput.h
  tSemaphore.h
  tStdLogOutput.h
  tStdErrorLogOutput.h
  tThread.h
)
ENDIF()

IF(WIN32)
  ICMAKER_ADD_SOURCES(
    SemaphoreImplWin32.cpp
    ThreadImplWin32.cpp
    )
  ICMAKER_ADD_HEADERS(
    SemaphoreImplWin32.h
    ThreadImplWin32.h
    )
ENDIF(WIN32)

IF(UNIX)
  # TODO: make this a FindFoo.cmake
  ICMAKER_LOCAL_CPPDEFINES(-D_IC_BUILDER_HAS_TIME_H_ -D_IC_BUILDER_HAS_SYS_TIME_H_)
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT Pthread)

  ICMAKER_ADD_SOURCES(
    stream_operators_impl_posix.cpp
    ThreadImplPosix.cpp
    UdpLogOutput.cpp
    )
  ICMAKER_ADD_HEADERS(
    stream_operators_impl_posix.h
    ThreadImplPosix.h
    UdpLogOutput.h
    )
ENDIF(UNIX)

IF(UNIX AND NOT APPLE)
  ICMAKER_ADD_SOURCES(SemaphoreImplPosix.cpp)
  ICMAKER_ADD_HEADERS(SemaphoreImplPosix.h)
ENDIF(UNIX AND NOT APPLE)

IF(UNIX AND APPLE)
  ICMAKER_ADD_SOURCES(SemaphoreImplDarwin.cpp)
  ICMAKER_ADD_HEADERS(SemaphoreImplDarwin.h)
ENDIF(UNIX AND APPLE)

IF(SQLite3_FOUND)
  ICMAKER_ADD_SOURCES(
    SQLiteLogDb.cpp
    SQLiteLogOutput.cpp
    )
  ICMAKER_ADD_HEADERS(
    SQLiteLogDb.h
    SQLiteLogOutput.h
    )
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
    SQLite3
    )
ENDIF(SQLite3_FOUND)

# @TODO:
# - pthread dependency on posix
# - lxrt
# - #ifdef ICL_CORE_QT_SUPPORT => QtCore dependency

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_LOGGING_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_LOGGING_)

ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  Boost_REGEX
)

ICMAKER_EXTERNAL_DEPENDENCIES(
  Boost_FILESYSTEM
  Boost_SYSTEM
)
ICMAKER_EXTERNAL_DEPENDENCIES(OPTIONAL
  Zlib
  Eigen2
  Eigen3
  )

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
)

IF(ANDROID)
  ICMAKER_SYSTEM_DEPENDENCIES(-llog)
ENDIF()

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(${icmaker_target})
