# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("icl_core_testsuite" IDE_FOLDER${ICL_CORE_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
    test_suite.cpp
    TestListener.cpp
)

ICMAKER_ADD_HEADERS(
    ImportExport.h
    test_suite.h
    TestListener.h
    )

ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_TESTSUITE_EXPORT_SYMBOLS)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_TESTSUITE_)
ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_logging
)

ICMAKER_EXTERNAL_DEPENDENCIES(
  CppUnit
)

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(${icmaker_target})
