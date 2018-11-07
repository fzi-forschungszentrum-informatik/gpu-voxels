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
# \date    2014-08-13
#
# Try to find FLIRTLib.  Once done, this will define:
#  FLIRTLib_FOUND:          System has FLIRTLib
#  FLIRTLib_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  FLIRTLib_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  FLIRTLib_DEFINITIONS:    Preprocessor definitions.
#  FLIRTLib_LIBRARIES:      only the libraries (w/o the '-l')
#  FLIRTLib_LDFLAGS:        all required linker flags
#  FLIRTLib_LDFLAGS_OTHER:  all other linker flags
#  FLIRTLib_CFLAGS:         all required cflags
#  FLIRTLib_CFLAGS_OTHER:   the other compiler flags
#  FLIRTLib_VERSION:        version of the module
#  FLIRTLib_PREFIX:         prefix-directory of the module
#  FLIRTLib_INCLUDEDIR:     include-dir of the module
#  FLIRTLib_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(FLIRTLib flirtlib
  HEADERS flirtlib/feature/InterestPoint.h
  LIBRARIES flirt_sensors flirt_sensorstream flirt_geometry flirt_feature flirt_utils
  HINTS /opt/local
  )
