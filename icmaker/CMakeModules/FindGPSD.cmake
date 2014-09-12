# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find GPSD.  Once done, this will define:
#  GPSD_FOUND:          System has GPSD
#  GPSD_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  GPSD_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  GPSD_DEFINITIONS:    Preprocessor definitions.
#  GPSD_LIBRARIES:      only the libraries (w/o the '-l')
#  GPSD_LDFLAGS:        all required linker flags
#  GPSD_LDFLAGS_OTHER:  all other linker flags
#  GPSD_CFLAGS:         all required cflags
#  GPSD_CFLAGS_OTHER:   the other compiler flags
#  GPSD_VERSION:        version of the module
#  GPSD_PREFIX:         prefix-directory of the module
#  GPSD_INCLUDEDIR:     include-dir of the module
#  GPSD_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(GPSD libgps
  HEADERS gps.h
  LIBRARIES gps
  HINTS /opt/local
  )
