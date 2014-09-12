# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find DC1394V2.  Once done, this will define:
#  DC1394V2_FOUND:          System has DC1394V2
#  DC1394V2_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  DC1394V2_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  DC1394V2_DEFINITIONS:    Preprocessor definitions.
#  DC1394V2_LIBRARIES:      only the libraries (w/o the '-l')
#  DC1394V2_LDFLAGS:        all required linker flags
#  DC1394V2_LDFLAGS_OTHER:  all other linker flags
#  DC1394V2_CFLAGS:         all required cflags
#  DC1394V2_CFLAGS_OTHER:   the other compiler flags
#  DC1394V2_VERSION:        version of the module
#  DC1394V2_PREFIX:         prefix-directory of the module
#  DC1394V2_INCLUDEDIR:     include-dir of the module
#  DC1394V2_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(DC1394V2 libdc1394-2
  HEADERS dc1394/control.h
  LIBRARIES dc1394
  )
