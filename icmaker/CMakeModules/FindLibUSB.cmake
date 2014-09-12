# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find LibUSB.  Once done, this will define:
#  LibUSB_FOUND:          System has LibUSB
#  LibUSB_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  LibUSB_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  LibUSB_DEFINITIONS:    Preprocessor definitions.
#  LibUSB_LIBRARIES:      only the libraries (w/o the '-l')
#  LibUSB_LDFLAGS:        all required linker flags
#  LibUSB_LDFLAGS_OTHER:  all other linker flags
#  LibUSB_CFLAGS:         all required cflags
#  LibUSB_CFLAGS_OTHER:   the other compiler flags
#  LibUSB_VERSION:        version of the module
#  LibUSB_PREFIX:         prefix-directory of the module
#  LibUSB_INCLUDEDIR:     include-dir of the module
#  LibUSB_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(LibUSB libusb
  HEADERS usb.h
  LIBRARIES usb
  )
