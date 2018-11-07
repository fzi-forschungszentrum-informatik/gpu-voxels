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

# - Try to find SerialConverter Driver LibFtd
# Once done this will define
#  ftd2xx_FOUND - System has ftd2xx
#  ftd2xx_INCLUDE_DIRS - The ftd2xx include directories
#  ftd2xx_LIBRARIES - link these to use ftd2xx

include(PrintLibraryStatus)
include(LibFindMacros)

find_path(ftd2xx_INCLUDE_DIR
  NAMES ftd2xx.h
  PATHS "/usr/include"
)

find_library(ftd2xx_LIBRARY
  NAMES libftd2xx.so
  PATHS "/usr/lib" "/usr/local/lib"
)


set(ftd2xx_PROCESS_INCLUDES ftd2xx_INCLUDE_DIR)
set(ftd2xx_PROCESS_LIBS ftd2xx_LIBRARY)
libfind_process(ftd2xx)


PRINT_LIBRARY_STATUS(ftd2xx
  DETAILS "[${ftd2xx_LIBRARIES}][${ftd2xx_INCLUDE_DIRS}]"
)
