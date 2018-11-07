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
include(PrintLibraryStatus)

  if(NOT OPENNI_ROOT AND ("ON" STREQUAL "ON"))
    set(OPENNI_INCLUDE_DIRS_HINT "/usr/include/ni")
    get_filename_component(OPENNI_LIBRARY_HINT "/usr/lib/libOpenNI.so" PATH)
  endif(NOT OPENNI_ROOT AND ("ON" STREQUAL "ON"))

  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_OPENNI libopenni)
  endif(PKG_CONFIG_FOUND)  
  find_path(OPENNI_INCLUDE_DIRS XnStatus.h
    HINTS ${PC_OPENNI_INCLUDEDIR} ${PC_OPENNI_INCLUDE_DIRS}
          "${OPENNI_ROOT}" "$ENV{OPENNI_ROOT}"
    PATHS "$ENV{OPEN_NI_INCLUDE}" "${OPENNI_INCLUDE_DIRS_HINT}"
    PATH_SUFFIXES include/openni Include)
  #add a hint so that it can find it without the pkg-config
  set(OPENNI_SUFFIX)
  if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(OPENNI_SUFFIX 64) 
  endif(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
  find_library(OPENNI_LIBRARY
    NAMES OpenNI64 OpenNI  
    HINTS ${PC_OPENNI_LIBDIR} ${PC_OPENNI_LIBRARY_DIRS}
          "${OPENNI_ROOT}" "$ENV{OPENNI_ROOT}"
    PATHS "$ENV{OPEN_NI_LIB${OPENNI_SUFFIX}}" "${OPENNI_LIBRARY_HINT}"
    PATH_SUFFIXES lib Lib Lib64)

  find_package_handle_standard_args(openni DEFAULT_MSG OPENNI_LIBRARY OPENNI_INCLUDE_DIRS)

  if(OPENNI_FOUND)
    get_filename_component(OPENNI_LIBRARY_PATH ${OPENNI_LIBRARY} PATH)
    set(OPENNI_LIBRARY_DIRS ${OPENNI_LIBRARY_PATH})
    set(OPENNI_LIBRARIES "${OPENNI_LIBRARY}")
  endif(OPENNI_FOUND)
