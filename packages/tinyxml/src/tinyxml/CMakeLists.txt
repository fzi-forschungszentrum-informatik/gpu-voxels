# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
ICMAKER_SET("tinyxml" IDE_FOLDER ${TINYXML_IDE_FOLDER})

ICMAKER_ADD_SOURCES(
  tinyxml.cpp
  tinystr.cpp
  tinyxmlerror.cpp
  tinyxmlparser.cpp
)

ICMAKER_ADD_HEADERS(
  tinystr.h
  tinyxml.h
)

ICMAKER_LOCAL_CPPDEFINES(-DTINYXML_EXPORT)
ICMAKER_GLOBAL_CPPDEFINES(-DTIXML_USE_STL -D_IC_BUILDER_TINYXML_)

ICMAKER_INCLUDE_DIRECTORIES(${TINYXML_INCLUDE_DIRS})

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(tinyxml)
