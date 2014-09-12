# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find PeakCan.  Once done, this will define:
#  PeakCan_FOUND:          System has PeakCan
#  PeakCan_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  PeakCan_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  PeakCan_DEFINITIONS:    Preprocessor definitions.
#  PeakCan_LIBRARIES:      only the libraries (w/o the '-l')
#  PeakCan_LDFLAGS:        all required linker flags
#  PeakCan_LDFLAGS_OTHER:  all other linker flags
#  PeakCan_CFLAGS:         all required cflags
#  PeakCan_CFLAGS_OTHER:   the other compiler flags
#  PeakCan_VERSION:        version of the module
#  PeakCan_PREFIX:         prefix-directory of the module
#  PeakCan_INCLUDEDIR:     include-dir of the module
#  PeakCan_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(PeakCan libpcan
  HEADERS libpcan.h
  LIBRARIES pcan
  DEFINE _IC_BUILDER_CAN_PEAK_
  )
