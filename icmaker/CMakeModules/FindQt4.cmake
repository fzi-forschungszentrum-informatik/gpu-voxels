# - Try to find Qt4 include dirs and libraries
#
# Please see the Documentation for Qt4 in the CMake Manual for details
# This module only forwards to the one included in cmake for compatibility
# reasons.

#  Based on FindBoost.cmake which is
#  Copyright (c) 2009      Andreas Pakulat <apaku@gmx.de>
#
#  Redistribution AND use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.

include(PrintLibraryStatus)

MACRO (_Qt4_ADJUST_INC_VARS basename)
  IF(QT_INCLUDE_DIR AND QT_${basename}_FOUND)
    SET(QT_${basename}_INCLUDE_DIRS ${QT_INCLUDE_DIR} CACHE FILEPATH "The Qt4 ${basename} include directory")
    #SET(QT_${basename}_DEFINITIONS "-D_IC_BUILDER_QT4_")
    MARK_AS_ADVANCED(
        QT_${basename}_INCLUDE_DIRS
    )
  ENDIF(QT_INCLUDE_DIR AND QT_${basename}_FOUND)
  # Make variables changeble to the advanced user
ENDMACRO (_Qt4_ADJUST_INC_VARS)

IF(QT4_FOUND)
   # in cache already
   SET(Qt4_FIND_QUIETLY TRUE)
ENDIF()

include(${CMAKE_ROOT}/Modules/FindQt4.cmake)

# Manually add the component include dirs
# The FindQt4.cmake module does not provide
# the components together with their include dirs
_Qt4_ADJUST_INC_VARS(QTCORE)
_Qt4_ADJUST_INC_VARS(QTGUI)
_Qt4_ADJUST_INC_VARS(QT3SUPPORT)
_Qt4_ADJUST_INC_VARS(QTASSISTANT)
_Qt4_ADJUST_INC_VARS(QTDBUS)
_Qt4_ADJUST_INC_VARS(QTDESIGNER)
_Qt4_ADJUST_INC_VARS(QTMOTIF)
_Qt4_ADJUST_INC_VARS(QTMULTIMEDIA)
_Qt4_ADJUST_INC_VARS(QTNETWORK)
_Qt4_ADJUST_INC_VARS(QTOPENGL)
_Qt4_ADJUST_INC_VARS(QTSQL)
_Qt4_ADJUST_INC_VARS(QTSVG)
_Qt4_ADJUST_INC_VARS(QTSCRIPT)
_Qt4_ADJUST_INC_VARS(QTSCRIPTTOOLS)
_Qt4_ADJUST_INC_VARS(QTTEST)
_Qt4_ADJUST_INC_VARS(QTUITOOLS)
_Qt4_ADJUST_INC_VARS(QTWEBKIT)
_Qt4_ADJUST_INC_VARS(QTXML)
_Qt4_ADJUST_INC_VARS(QTXMLPATTERNS)
_Qt4_ADJUST_INC_VARS(PHONON)
_Qt4_ADJUST_INC_VARS(QTDECLARATIVE)

#FIND_PACKAGE_HANDLE_STANDARD_ARGS is called by FindQt4.cmake itself
IF(NOT QT4_FOUND )
    PRINT_LIBRARY_STATUS(Qt4
          COMPONENTS "${Qt4_FIND_COMPONENTS}"
    )
ENDIF()
