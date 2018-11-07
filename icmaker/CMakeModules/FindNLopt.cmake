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
# \author  Sebastian Klemm <klemm@fzi.de>
# \date    2014-10-29
#
# Try to find NLopt (http://ab-initio.mit.edu/wiki/index.php/NLopt)
# Once done, this will define:
#  NLopt_FOUND:          System has NLopt
#  NLopt_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  NLopt_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  NLopt_DEFINITIONS:    Preprocessor definitions.
#  NLopt_LIBRARIES:      only the libraries (w/o the '-l')
#  NLopt_LDFLAGS:        all required linker flags
#  NLopt_LDFLAGS_OTHER:  all other linker flags
#  NLopt_CFLAGS:         all required cflags
#  NLopt_CFLAGS_OTHER:   the other compiler flags
#  NLopt_VERSION:        version of the module
#  NLopt_PREFIX:         prefix-directory of the module
#  NLopt_INCLUDEDIR:     include-dir of the module
#  NLopt_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(NLopt nlopt
  HEADERS nlopt.h
  HEADER_PATHS "/usr/include"
  DEFINE _IC_BUILDER_NLOPT_
  )
