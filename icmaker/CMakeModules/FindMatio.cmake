# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Matio.  Once done, this will define:
#  Matio_FOUND:          System has Matio
#  Matio_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Matio_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Matio_DEFINITIONS:    Preprocessor definitions.
#  Matio_LIBRARIES:      only the libraries (w/o the '-l')
#  Matio_LDFLAGS:        all required linker flags
#  Matio_LDFLAGS_OTHER:  all other linker flags
#  Matio_CFLAGS:         all required cflags
#  Matio_CFLAGS_OTHER:   the other compiler flags
#  Matio_VERSION:        version of the module
#  Matio_PREFIX:         prefix-directory of the module
#  Matio_INCLUDEDIR:     include-dir of the module
#  Matio_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Matio matio
  HEADERS matio.h
  LIBRARIES matio
  )
