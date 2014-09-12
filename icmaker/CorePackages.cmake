# This file is included from CMakeLists.txt and contains packages used during built
# CorePackages.cmake is provided by icmaker
#
# If you want to specify other packages, please use UserPackages.cmake file.

# core:
FIND_PACKAGE(Boost 1.42.0 COMPONENTS date_time filesystem graph iostreams system regex signals
                                     unit_test_framework thread program_options serialization)
                                     # Additional components:
                                     # math_c99 math_c99f math_c99l math_tr1 math_tr1f math_tr1l
                                     # prg_exec_monitor random
                                     # thread wave wserialization
FIND_PACKAGE(Boost 1.42.0 COMPONENTS python)
FIND_PACKAGE(Boost 1.42.0)
FIND_PACKAGE(CppUnit)
FIND_PACKAGE(Crypt)
FIND_PACKAGE(DL)
FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
  FIND_PACKAGE(HTMLHelp)
ENDIF(DOXYGEN_FOUND)
FIND_PACKAGE(Eigen3)
FIND_PACKAGE(GLUT)
  PRINT_LIBRARY_STATUS(GLUT DETAILS "[${GLUT_LIBRARIES}]")
FIND_PACKAGE(GSL)
FIND_PACKAGE(Iconv)
FIND_PACKAGE(JPEG)
FIND_PACKAGE(LibArchive)
FIND_PACKAGE(libwebsockets)
FIND_PACKAGE(LibXml2)
FIND_PACKAGE(Ltdl)
FIND_PACKAGE(Newmat)
FIND_PACKAGE(OpenGL)
  PRINT_LIBRARY_STATUS(OPENGL DETAILS "[${OPENGL_LIBRARIES}]")
FIND_PACKAGE(Pthread)
IF (NOT ICMAKER_USE_QT3)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtWebKit)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtDBus)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS Phonon)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtSvg)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtOpenGL)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS Qt3Support)
  FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtCore QtGui QtDeclarative QtXml QtNetwork)
ENDIF()
FIND_PACKAGE(SCIP)
FIND_PACKAGE(Speechd)
FIND_PACKAGE(SQLite3)
FIND_PACKAGE(SSL)
FIND_PACKAGE(Threads)
FIND_PACKAGE(Xsd)
FIND_PACKAGE(Zlib)
FIND_PACKAGE(Inventor)
FIND_PACKAGE(Spacenav)

IF (ICMAKER_USE_PERCEPTION_PACKAGES)
  # perception:
  MESSAGE(STATUS "=== Searching Perception packages ===")
  FIND_PACKAGE(OpenCV)
    PRINT_LIBRARY_STATUS(OpenCV DETAILS "[${OpenCV_LIBRARIES}][${OpenCV_INCLUDE_DIRS}]")
  FIND_PACKAGE(PCL QUIET)
ENDIF()

IF (ICMAKER_USE_PARALLEL_PACKAGES)
  # parallel:
  MESSAGE(STATUS "=== Searching Parallel packages ===")
  FIND_PACKAGE(CUDA)
  FIND_PACKAGE(OpenCL)
  FIND_PACKAGE(OpenMP)
  FIND_PACKAGE(NPP)
ENDIF()


IF(ICMAKER_USE_MAPS_PACKAGES)
  # maps:
  MESSAGE(STATUS "=== Searching Maps packages ===")
  FIND_PACKAGE(GDal)
  FIND_PACKAGE(Marble)
ENDIF()

IF(ICMAKER_USE_HARDWARE_PACKAGES)
# hardware:
  MESSAGE(STATUS "=== Searching Hardware packages ===")
  FIND_PACKAGE(DC1394V2)
  FIND_PACKAGE(FTD2XX)
  FIND_PACKAGE(GPSD)
  FIND_PACKAGE(Freenect)
  FIND_PACKAGE(LibUSB)
  FIND_PACKAGE(MesaSR)
  FIND_PACKAGE(PeakCan)
  FIND_PACKAGE(PmdSDK2)
ENDIF()

IF(ICMAKER_USE_VISUALIZATION_PACKAGES)
  # visualization:
  MESSAGE(STATUS "=== Searching Visualization packages ===")
  FIND_PACKAGE(Coin3D)
  FIND_PACKAGE(osg)
  IF(OSG_FOUND)
    FIND_PACKAGE(osgAnimation)
    FIND_PACKAGE(osgDB)
    FIND_PACKAGE(osgFX)
    FIND_PACKAGE(osgGA)
    FIND_PACKAGE(osgIntrospection)
    FIND_PACKAGE(osgManipulator)
    FIND_PACKAGE(osgParticle)
    FIND_PACKAGE(osgProducer)
    FIND_PACKAGE(osgShadow)
    FIND_PACKAGE(osgSim)
    FIND_PACKAGE(osgTerrain)
    FIND_PACKAGE(osgText)
    FIND_PACKAGE(osgUtil)
    FIND_PACKAGE(osgViewer)
    FIND_PACKAGE(osgVolume)
    FIND_PACKAGE(osgWidget)
    FIND_PACKAGE(OpenThreads)
  ENDIF()
ENDIF()

IF(ICMAKER_USE_COMMUNICATION_PACKAGES)
  # communication:
  MESSAGE(STATUS "=== Searching Communication packages ===")
  FIND_PACKAGE(KogmoRtdb)
  FIND_PACKAGE(OpenSpliceDDS)
  FIND_PACKAGE(SimDcxx)
ENDIF()

IF(ICMAKER_USE_OTHER_PACKAGES)
  # others:
  MESSAGE(STATUS "=== Searching Other packages ===")
  FIND_PACKAGE(Antlr3)
  FIND_PACKAGE(CoreFoundation)
  FIND_PACKAGE(LibDAI)
  FIND_PACKAGE(NetSNMP) # iboss
  FIND_PACKAGE(PythonLibs)
  FIND_PACKAGE(SVM)
  FIND_PACKAGE(Xdo)
  FIND_PACKAGE(DXFLib)
ENDIF()

IF(ICMAKER_USE_DEPRECATED_PACKAGES)
  # deprecated:
  MESSAGE(STATUS "=== Searching Deprecated packages ===")
  FIND_PACKAGE(Eigen2)
  IF (ICMAKER_USE_QT3)
    FIND_PACKAGE(Qt3)
    ADD_DEFINITIONS(-D_IC_BUILDER_QT_3_)
  ENDIF()
ENDIF()

IF(ICMAKER_USE_EXTRA_PACKAGES)
INCLUDE(ExtraPackages.cmake OPTIONAL)
ENDIF()
