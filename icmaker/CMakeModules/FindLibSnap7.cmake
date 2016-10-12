# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Andreas Hermann <hermann@fzi.de>
# \date    2015-02-24
#
# Try to find LibSnap7.  Once done, this will define:
#  LibSnap7_FOUND:          System has LibSnap7
#  LibSnap7_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  LibSnap7_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  LibSnap7_DEFINITIONS:    Preprocessor definitions.
#  LibSnap7_LIBRARIES:      only the libraries (w/o the '-l')
#  LibSnap7_LDFLAGS:        all required linker flags
#  LibSnap7_LDFLAGS_OTHER:  all other linker flags
#  LibSnap7_CFLAGS:         all required cflags
#  LibSnap7_CFLAGS_OTHER:   the other compiler flags
#  LibSnap7_VERSION:        version of the module
#  LibSnap7_PREFIX:         prefix-directory of the module
#  LibSnap7_INCLUDEDIR:     include-dir of the module
#  LibSnap7_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(LibSnap7 libsnap7
  LIBRARIES snap7
  )
