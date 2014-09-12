# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Xdo.  Once done, this will define:
#  Xdo_FOUND:          System has Xdo
#  Xdo_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Xdo_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Xdo_DEFINITIONS:    Preprocessor definitions.
#  Xdo_LIBRARIES:      only the libraries (w/o the '-l')
#  Xdo_LDFLAGS:        all required linker flags
#  Xdo_LDFLAGS_OTHER:  all other linker flags
#  Xdo_CFLAGS:         all required cflags
#  Xdo_CFLAGS_OTHER:   the other compiler flags
#  Xdo_VERSION:        version of the module
#  Xdo_PREFIX:         prefix-directory of the module
#  Xdo_INCLUDEDIR:     include-dir of the module
#  Xdo_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Xdo xdo
  HEADERS xdo.h
  LIBRARIES xdo
  )
