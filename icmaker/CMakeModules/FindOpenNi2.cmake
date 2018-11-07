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

include(PrintLibraryStatus)
include(LibFindMacros)
include(FindPkgConfig)

  if(NOT OPENNI2_ROOT AND ("ON" STREQUAL "ON"))
    get_filename_component(OPENNI2_LIBRARY_HINT "OPENNI_LIBRARY-NOTFOUND" PATH)
  endif(NOT OPENNI2_ROOT AND ("ON" STREQUAL "ON"))

  set(OPENNI2_SUFFIX)
  if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OPENNI2_SUFFIX 64)
  endif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)

  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_OPENNI2 libopenni2)
  endif(PKG_CONFIG_FOUND)

  find_path(OPENNI2_INCLUDE_DIRS OpenNI.h
          HINTS /usr/include/openni2 /usr/include/ni2
          PATHS "$ENV{OPENNI2_INCLUDE${OPENNI2_SUFFIX}}"
          PATH_SUFFIXES openni openni2 include Include)

  find_library(OPENNI2_LIBRARY
             NAMES OpenNI2      # No suffix needed on Win64
             HINTS /usr/lib
             PATHS "$ENV{OPENNI2_LIB${OPENNI2_SUFFIX}}"
             PATH_SUFFIXES lib Lib Lib64)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(OpenNI2 DEFAULT_MSG OPENNI2_LIBRARY OPENNI2_INCLUDE_DIRS)

  if(OPENNI2_FOUND)
    get_filename_component(OPENNI_LIBRARY_PATH ${OPENNI2_LIBRARY} PATH)
    set(OPENNI2_LIBRARY_DIRS ${OPENNI2_LIBRARY_PATH})
    set(OPENNI2_LIBRARIES "${OPENNI2_LIBRARY}")
    set(OPENNI2_REDIST_DIR $ENV{OPENNI2_REDIST${OPENNI2_SUFFIX}})
  endif(OPENNI2_FOUND)
