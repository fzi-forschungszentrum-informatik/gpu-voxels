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

# - Try to find libjsoncpp
# Once done, this will define
#
# JSONCPP_FOUND - system has libjsoncpp
# JSONCPP_INCLUDE_DIRS - the libjsoncpp include directories
# JSONCPP_LIBRARIES - link these to use libjsoncpp

include(LibFindMacros)

IF (UNIX)
  libfind_pkg_check_modules(JSONCPP_PKGCONF jsoncpp)

  find_path(JSONCPP_INCLUDE_DIR
    NAMES json/json.h
    PATHS ${JSONCPP_ROOT}/include ${JSONCPP_PKGCONF_INCLUDE_DIRS}
    )

  find_library(JSONCPP_LIBRARY
    NAMES jsoncpp
    PATHS ${JSONCPP_ROOT}/lib ${JSONCPP_PKGCONF_LIBRARY_DIRS}
    )
ELSEIF (WIN32)
  find_path(JSONCPP_INCLUDE_DIR
    NAMES json/json.h
    PATHS ${JSONCPP_ROOT}/include ${CMAKE_INCLUDE_PATH}
    )
  find_library(JSONCPP_LIBRARY
    NAMES libjsoncpp
    PATHS ${JSONCPP_ROOT}/lib ${CMAKE_LIB_PATH}
    )
ENDIF()

set(JSONCPP_PROCESS_INCLUDES JSONCPP_INCLUDE_DIR JSONCPP_INCLUDE_DIRS)
set(JSONCPP_PROCESS_LIBS JSONCPP_LIBRARY JSONCPP_LIBRARIES)
libfind_process(JSONCPP)
