# -- BEGIN LICENSE BLOCK ----------------------------------------------
# Copyright (c) 2018, FZI Forschungszentrum Informatik
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- END LICENSE BLOCK ------------------------------------------------

# This file is included from CMakeLists.txt and contains packages used during built
# CorePackages.cmake is provided by icmaker
#
# If you want to specify other packages, please use UserPackages.cmake file.

# the default value is off!
option(USE_Qt5 "Use Qt5 for the build" OFF)

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
IF(WIN32)
  FIND_PACKAGE(DlfcnWin32)
ENDIF()
FIND_PACKAGE(GLUT)
  PRINT_LIBRARY_STATUS(GLUT DETAILS "[${GLUT_LIBRARIES}]")
FIND_PACKAGE(GSL)
FIND_PACKAGE(Iconv)
FIND_PACKAGE(JPEG)
FIND_PACKAGE(LibArchive)
FIND_PACKAGE(libwebsockets)
FIND_PACKAGE(LibXml2)
FIND_PACKAGE(Ltdl)
FIND_PACKAGE(Ncomrx)
FIND_PACKAGE(Newmat)
FIND_PACKAGE(OpenGL)
  PRINT_LIBRARY_STATUS(OPENGL DETAILS "[${OPENGL_LIBRARIES}]")
FIND_PACKAGE(Pthread)
IF(USE_Qt5)
  FIND_PACKAGE(Qt5IcMaker)
ELSE()
  IF (NOT ICMAKER_USE_QT3)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtWebKit)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtDBus)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS Phonon)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtSvg)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtOpenGL)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS Qt3Support)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtDeclarative)
    FIND_PACKAGE(Qt4 4.4.0 COMPONENTS QtCore QtGui QtXml QtNetwork)
  ENDIF()
ENDIF()
FIND_PACKAGE(SCIP)
FIND_PACKAGE(Speechd)
FIND_PACKAGE(SQLite3)
FIND_PACKAGE(SSL)
FIND_PACKAGE(Tcmalloc)
FIND_PACKAGE(Threads)

FIND_PACKAGE(TinyXML)
IF (TINYXML_FOUND)
  ADD_DEFINITIONS(-DTIXML_USE_STL)
ENDIF()

FIND_PACKAGE(Xsd)
FIND_PACKAGE(Zlib)
FIND_PACKAGE(Inventor)
FIND_PACKAGE(Spacenav)
FIND_PACKAGE(X11)

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
  FIND_PACKAGE(PythonInterp)
  FIND_PACKAGE(SVM)
  FIND_PACKAGE(Xdo)
  FIND_PACKAGE(DXFLib)
  FIND_PACKAGE(JNI)
  FIND_PACKAGE(PerlLibs)
  FIND_PACKAGE(SWIG)
  FIND_PACKAGE(ROS)
  FIND_PACKAGE(GTSAM)
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
