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

# - Try to find FLandmarks
# Once done, this will define
#
#  FLandmarks_FOUND - system has GDal
#  FLandmarks_INCLUDE_DIR - the GDal include directories
#  FLandmarks_LIBRARY - link these to use GDal

include(PrintLibraryStatus)
include(LibFindMacros)

if ( FLandmarks_FOUND )
   # in cache already
   SET( FLandmarks_FIND_QUIETLY TRUE )
endif ()

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(FLandmarks_PKGCONF flandmarks)

# Include dir
find_path(FLandmarks_INCLUDE_DIR
  NAMES flandmark_detector.h
  PATHS ${FLandmarks_PKGCONF_INCLUDE_DIRS} "/usr/include/flandmarks" "/opt/local/include"
)

# Finally the library itself
find_library(FLandmarks_LIBRARY
  NAMES flandmark_shared
  PATHS ${FLandmarks_PKGCONF_LIBRARY_DIRS} "/opt/local/lib"
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(FLandmarks_PROCESS_INCLUDES FLandmarks_INCLUDE_DIR)
set(FLandmarks_PROCESS_LIBS FLandmarks_LIBRARY)
libfind_process(FLandmarks)

PRINT_LIBRARY_STATUS(FLandmarks
  DETAILS "[${FLandmarks_LIBRARIES}][${FLandmarks_INCLUDE_DIRS}]"
)
