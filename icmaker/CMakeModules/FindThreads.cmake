# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

# Extend the CMake FindThreads module.

include(${CMAKE_ROOT}/Modules/FindThreads.cmake)

IF (Threads_FOUND)
  IF (CMAKE_USE_PTHREADS_INIT)
    SET(Threads_DEFINITIONS -D_IC_BUILDER_PTHREAD_)
  ENDIF (CMAKE_USE_PTHREADS_INIT)
  SET(Threads_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
ENDIF (Threads_FOUND)
