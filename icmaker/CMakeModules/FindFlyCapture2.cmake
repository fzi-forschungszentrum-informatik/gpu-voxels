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

# - Try to find FlyCapture2
# Once done, this will define
#
#  FlyCapture2_FOUND - system has FlyCapture2
#  FlyCapture2_INCLUDE_DIRS - the FlyCapture2 include directories
#  FlyCapture2_LIBRARIES - link these to use FlyCapture2

IF( FlyCapture2_FOUND )
   # in cache already
   SET( FlyCapture2_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Include dir
find_path(FlyCapture2_INCLUDE_DIR
  NAMES flycapture/Camera.h
)

# Finally the library itself
find_library(FlyCapture2_LIBRARY
  NAMES flycapture
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(FlyCapture2_PROCESS_INCLUDES FlyCapture2_INCLUDE_DIR)
set(FlyCapture2_PROCESS_LIBS FlyCapture2_LIBRARY)
libfind_process(FlyCapture2)

PRINT_LIBRARY_STATUS(FlyCapture2
  DETAILS "[${FlyCapture2_LIBRARIES}][${FlyCapture2_INCLUDE_DIRS}]"
)
