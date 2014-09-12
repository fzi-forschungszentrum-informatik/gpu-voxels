# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Cantools.  Once done, this will define:
#  Cantools_FOUND:          System has Cantools
#  Cantools_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Cantools_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Cantools_DEFINITIONS:    Preprocessor definitions.
#  Cantools_LIBRARIES:      only the libraries (w/o the '-l')
#  Cantools_LDFLAGS:        all required linker flags
#  Cantools_LDFLAGS_OTHER:  all other linker flags
#  Cantools_CFLAGS:         all required cflags
#  Cantools_CFLAGS_OTHER:   the other compiler flags
#  Cantools_VERSION:        version of the module
#  Cantools_PREFIX:         prefix-directory of the module
#  Cantools_INCLUDEDIR:     include-dir of the module
#  Cantools_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Cantools cantools
  HEADERS dbcModel.h
  LIBRARIES candbc
  HINTS /opt/local
  )
