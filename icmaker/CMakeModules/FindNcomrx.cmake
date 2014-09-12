# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Ncomrx.  Once done, this will define:
#  Ncomrx_FOUND:          System has Ncomrx
#  Ncomrx_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Ncomrx_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Ncomrx_DEFINITIONS:    Preprocessor definitions.
#  Ncomrx_LIBRARIES:      only the libraries (w/o the '-l')
#  Ncomrx_LDFLAGS:        all required linker flags
#  Ncomrx_LDFLAGS_OTHER:  all other linker flags
#  Ncomrx_CFLAGS:         all required cflags
#  Ncomrx_CFLAGS_OTHER:   the other compiler flags
#  Ncomrx_VERSION:        version of the module
#  Ncomrx_PREFIX:         prefix-directory of the module
#  Ncomrx_INCLUDEDIR:     include-dir of the module
#  Ncomrx_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Ncomrx ncomrx
  HEADERS NComRxC.h
  LIBRARIES ncomrx
  )
