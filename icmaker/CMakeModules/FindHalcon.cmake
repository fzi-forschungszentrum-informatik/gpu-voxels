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
# \author  Steffen Rühl <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Halcon.  Once done, this will define:
#  Halcon_FOUND:          System has Halcon
#  Halcon_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Halcon_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Halcon_DEFINITIONS:    Preprocessor definitions.
#  Halcon_LIBRARIES:      only the libraries (w/o the '-l')
#  Halcon_LDFLAGS:        all required linker flags
#  Halcon_LDFLAGS_OTHER:  all other linker flags
#  Halcon_CFLAGS:         all required cflags
#  Halcon_CFLAGS_OTHER:   the other compiler flags
#  Halcon_VERSION:        version of the module
#  Halcon_PREFIX:         prefix-directory of the module
#  Halcon_INCLUDEDIR:     include-dir of the module
#  Halcon_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------


include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Halcon halcon
  HEADER_PATHS "/opt/halcon/include" "/opt/halcon11/include"
  HEADERS HalconC.h
  LIBRARY_PATHS "/opt/halcon/lib/x64-linux" "/opt/halcon/lib/x86sse2-linux2.4-gcc40" "/opt/halcon11/lib/x64-linux2.4-gcc40"
  LIBRARIES halconcpp halcon

  )

# Add a subdirectory needed by halcon.
if (Halcon_FOUND)
  LIST(GET Halcon_INCLUDE_DIRS 0 dir)
  SET(subdir "${dir}/halconcpp")
  LIST(APPEND Halcon_INCLUDE_DIRS ${subdir})
endif ()
