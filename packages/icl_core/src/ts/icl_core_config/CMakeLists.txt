# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("ts_icl_core_config" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  ts_main.cpp
  ts_BatchGet.cpp
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
  icl_core_config
  icl_core_logging
  )

ICMAKER_BUILD_TEST()
