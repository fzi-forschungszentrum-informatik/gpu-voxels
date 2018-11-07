# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

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
