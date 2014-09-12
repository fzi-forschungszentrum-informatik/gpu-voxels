# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("uls" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  UdpLoggingServer.cpp
  uls.cpp
  )

ICMAKER_ADD_HEADERS(
  UdpLoggingServer.h
  )

ICMAKER_ADD_MOC_HEADERS(
  UdpLoggingServer.h
  )

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  )

ICMAKER_EXTERNAL_DEPENDENCIES(
  QT_QTCORE
  QT_QTNETWORK
  QT_QTSQL
  )

ICMAKER_BUILD_PROGRAM()
