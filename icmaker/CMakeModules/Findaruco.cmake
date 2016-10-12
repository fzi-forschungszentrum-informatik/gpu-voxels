# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Florian Kuhnt <kuhnt@fzi.de>
# \date    2016-03-15
#
# Try to find aruco.  Once done, this will define:
#  aruco_FOUND:          System has aruco
#  aruco_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  aruco_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  aruco_DEFINITIONS:    Preprocessor definitions.
#  aruco_LIBRARIES:      only the libraries (w/o the '-l')
#  aruco_LDFLAGS:        all required linker flags
#  aruco_LDFLAGS_OTHER:  all other linker flags
#  aruco_CFLAGS:         all required cflags
#  aruco_CFLAGS_OTHER:   the other compiler flags
#  aruco_VERSION:        version of the module
#  aruco_PREFIX:         prefix-directory of the module
#  aruco_INCLUDEDIR:     include-dir of the module
#  aruco_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(aruco aruco
  HINTS /usr/local/include/src
  HEADERS aruco/aruco.h
  LIBRARIES aruco
  )
