# Try to find SoQt
# Once done this will define
# From http://liris.cnrs.fr/dgtal/
#
# SoQt_FOUND        - system has SoQt - needs Coin3D - Open Inventor
# SoQt_INCLUDE_DIR  - where the SoQt include directory can be found
# SoQt_LIBRARIES      - Link this to use SoQt
#
 

IF (WIN32)
  IF (CYGWIN)

    FIND_PATH(SoQt_INCLUDE_DIR Inventor/Qt/SoQt.h)

    FIND_LIBRARY(SoQt_LIBRARIES SoQt)

  ELSE (CYGWIN)
    message("[xx] Unchecked system." )
    FIND_PATH(SoQt_INCLUDE_DIR Inventor/Qt/SoQt.h
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\SoQt\\2;Installation Path]/include"
    )

    FIND_LIBRARY(SoQt_LIBRARY_DEBUG soqtd
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\SoQt\\2;Installation Path]/lib"
    )

    FIND_LIBRARY(SoQt_LIBRARY_RELEASE soqt
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\SoQt\\2;Installation Path]/lib"
    )

    IF (SoQt_LIBRARY_DEBUG AND SoQt_LIBRARY_RELEASE)
      SET(SoQt_LIBRARIES optimized ${SoQt_LIBRARY_RELEASE}
                         debug ${SoQt_LIBRARY_DEBUG})
    ELSE (SoQt_LIBRARY_DEBUG AND SoQt_LIBRARY_RELEASE)
      IF (SoQt_LIBRARY_DEBUG)
        SET (SoQt_LIBRARIES ${SoQt_LIBRARY_DEBUG})
      ENDIF (SoQt_LIBRARY_DEBUG)
      IF (SoQt_LIBRARY_RELEASE)
        SET (SoQt_LIBRARIES ${SoQt_LIBRARY_RELEASE})
      ENDIF (SoQt_LIBRARY_RELEASE)
    ENDIF (SoQt_LIBRARY_DEBUG AND SoQt_LIBRARY_RELEASE)

    IF (SoQt_LIBRARY)
      ADD_DEFINITIONS ( -DSoQt_NOT_DLL )
    ENDIF (SoQt_LIBRARY)

  ENDIF (CYGWIN)

ELSE (WIN32)
  IF(APPLE)
    FIND_PATH(SoQt_INCLUDE_DIR Inventor/Qt/SoQt.h
     /Library/Frameworks/Inventor.framework/Headers 
    )
    FIND_LIBRARY(SoQt_LIBRARIES SoQt
      /Library/Frameworks/Inventor.framework/Libraries
    )   
    SET(SoQt_LIBRARIES "-framework SoQt" CACHE STRING "SoQt library for OSX")
  ELSE(APPLE)
       FIND_PATH(SoQt_INCLUDE_DIR Inventor/Qt/SoQt.h)
       FIND_LIBRARY(SoQt_LIBRARIES SoQt)
  ENDIF(APPLE)

ENDIF (WIN32)

# handle the QUIETLY and REQUIRED arguments and set SoQt_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SoQt DEFAULT_MSG SoQt_LIBRARIES SoQt_INCLUDE_DIR)

MARK_AS_ADVANCED(SoQt_INCLUDE_DIR SoQt_LIBRARIES )
