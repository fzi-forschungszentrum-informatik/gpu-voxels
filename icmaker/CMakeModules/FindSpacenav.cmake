# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Spacenav.  Once done, this will define:
#  Spacenav_FOUND:          System has Spacenav
#  Spacenav_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Spacenav_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Spacenav_DEFINITIONS:    Preprocessor definitions.
#  Spacenav_LIBRARIES:      only the libraries (w/o the '-l')
#  Spacenav_LDFLAGS:        all required linker flags
#  Spacenav_LDFLAGS_OTHER:  all other linker flags
#  Spacenav_CFLAGS:         all required cflags
#  Spacenav_CFLAGS_OTHER:   the other compiler flags
#  Spacenav_VERSION:        version of the module
#  Spacenav_PREFIX:         prefix-directory of the module
#  Spacenav_INCLUDEDIR:     include-dir of the module
#  Spacenav_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Spacenav spnav
  HEADERS spnav.h
  LIBRARIES spnav
  )
