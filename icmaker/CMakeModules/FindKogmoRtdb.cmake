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

# - Try to find KogmoRtdb
# Once done, this will define
#
#  KogmoRtdb_FOUND - system has KogmoRtdb
#  KogmoRtdb_INCLUDE_DIRS - the KogmoRtdb include directories
#  KogmoRtdb_LIBRARIES - link these to use KogmoRtdb

IF( KogmoRtdb_FOUND )
   # in cache already
   SET( KogmoRtdb_FIND_QUIETLY TRUE )
ENDIF()

include(PrintLibraryStatus)
include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(KogmoRtdb_PKGCONF libkogmo_rtdb)

# Include dir
find_path(KogmoRtdb_INCLUDE_DIR
  NAMES kogmo_rtdb.hxx
  PATHS ${KogmoRtdb_PKGCONF_INCLUDE_DIRS} "/usr/include/kogmo_rtdb"
)

find_path(KogmoObjects_INCLUDE_DIR
  NAMES kogmo_rtdb_obj_defs.h
  PATHS ${KogmoRtdb_PKGCONF_INCLUDE_DIRS} "/usr/include/kogmo_objects"
)

# Finally the library itself
find_library(KogmoRtdb_LIBRARY
  NAMES kogmo_rtdb
  PATHS ${KogmoRtdb_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(KogmoRtdb_PROCESS_INCLUDES KogmoRtdb_INCLUDE_DIR KogmoObjects_INCLUDE_DIR)
set(KogmoRtdb_PROCESS_LIBS KogmoRtdb_LIBRARY)
libfind_process(KogmoRtdb)

PRINT_LIBRARY_STATUS(KogmoRtdb
  DETAILS "[${KogmoRtdb_LIBRARIES}][${KogmoRtdb_INCLUDE_DIRS}]"
)

