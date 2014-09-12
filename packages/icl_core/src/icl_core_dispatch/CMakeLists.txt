# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_dispatch" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  CallbackOperation.cpp
  Logging.cpp
  Operation.cpp
  PeriodicDispatchThread.cpp
  )

ICMAKER_ADD_HEADERS(
  CallbackOperation.h
  Logging.h
  ImportExport.h
  Operation.h
  PeriodicDispatchThread.h
  )

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_DISPATCH_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_DISPATCH_)
ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_logging
  icl_core_thread
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  Boost
  )

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(${icmaker_target})
