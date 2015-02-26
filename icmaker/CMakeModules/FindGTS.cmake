# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find GTS.  Once done, this will define:
#  GTS_FOUND:          System has GTS
#  GTS_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  GTS_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  GTS_DEFINITIONS:    Preprocessor definitions.
#  GTS_LIBRARIES:      only the libraries (w/o the '-l')
#  GTS_LDFLAGS:        all required linker flags
#  GTS_LDFLAGS_OTHER:  all other linker flags
#  GTS_CFLAGS:         all required cflags
#  GTS_CFLAGS_OTHER:   the other compiler flags
#  GTS_VERSION:        version of the module
#  GTS_PREFIX:         prefix-directory of the module
#  GTS_INCLUDEDIR:     include-dir of the module
#  GTS_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(GTS gts
  HEADERS gts.h
  LIBRARIES gts
  )
