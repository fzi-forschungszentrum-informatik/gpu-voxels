# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-12-09
#
# Try to find Tcmalloc.  Once done, this will define:
# Tcmalloc_INCLUDE_DIR: Where to find Tcmalloc.h, etc.
# Tcmalloc_LIBRARIES:   List of libraries when using Tcmalloc.
# Tcmalloc_CFLAGS:      If GCC is used, sets flags so GCC doesn't make
#                       assumptions about using its own alloc
#                       routines.
# Tcmalloc_CXXFLAGS:    If GCC is used, sets flags so GCC doesn't make
#                       assumptions about using its own alloc
#                       routines.
# Tcmalloc_FOUND:       True if Tcmalloc was found.
#
# By setting ICMAKER_USE_TCMALLOC=True globally, every executable will
# link against tcmalloc and thus support heap profiling and leak
# checking using Google's perftools.  Alternatively you may instrument
# an individual program by adding a library dependency on Tcmalloc.
#----------------------------------------------------------------------

include(PrintLibraryStatus)

IF (Tcmalloc_INCLUDE_DIR)
  set(Tcmalloc_FIND_QUIETLY TRUE)
ENDIF ()

set(Tcmalloc_INCLUDE_HINT)
set(Tcmalloc_LIBRARY_HINT)
IF (Tcmalloc_ROOT)
  set(Tcmalloc_INCLUDE_HINT "${Tcmalloc_ROOT}/include")
  set(Tcmalloc_LIBRARY_HINT "${Tcmalloc_ROOT}/lib")
ENDIF ()

find_path(Tcmalloc_INCLUDE_DIR google/heap-checker.h
  ${Tcmalloc_INCLUDE_HINT}
  /usr/local/include
  /usr/include
  )

find_library(Tcmalloc_LIBRARY
  NAMES tcmalloc
  PATHS
  ${Tcmalloc_LIBRARY_HINT}
  /usr/local/lib
  /usr/lib
  )

IF (Tcmalloc_INCLUDE_DIR AND Tcmalloc_LIBRARY)
  set(Tcmalloc_FOUND TRUE)
  set(Tcmalloc_LIBRARIES ${Tcmalloc_LIBRARY})
ELSE ()
  set(Tcmalloc_FOUND FALSE)
  set(Tcmalloc_LIBRARIES)
ENDIF ()

IF (Tcmalloc_FOUND)
  # Add gcc-specific flags.
  IF (CMAKE_COMPILER_IS_GNUXX OR CMAKE_COMPILER_IS_GNUCC)
    set(Tcmalloc_FLAGS "-fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free" CACHE INTERNAL "")
  ENDIF ()
ELSE ()
  IF (Tcmalloc_FIND_REQUIRED)
    message(FATAL_ERROR "Could NOT find Tcmalloc library")
  ENDIF ()
  set(Tcmalloc_FLAGS "" CACHE INTERNAL "")
ENDIF ()

PRINT_LIBRARY_STATUS(Tcmalloc
  DETAILS "[${Tcmalloc_LIBRARY}][${Tcmalloc_INCLUDE_DIR}][${Tcmalloc_FLAGS}]"
  )

mark_as_advanced(
  Tcmalloc_LIBRARY
  Tcmalloc_INCLUDE_DIR
  Tcmalloc_FLAGS
  )
