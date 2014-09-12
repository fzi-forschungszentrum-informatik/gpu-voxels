# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find LibDAI.  Once done, this will define:
#  LibDAI_FOUND:          System has LibDAI
#  LibDAI_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  LibDAI_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  LibDAI_DEFINITIONS:    Preprocessor definitions.
#  LibDAI_LIBRARIES:      only the libraries (w/o the '-l')
#  LibDAI_LDFLAGS:        all required linker flags
#  LibDAI_LDFLAGS_OTHER:  all other linker flags
#  LibDAI_CFLAGS:         all required cflags
#  LibDAI_CFLAGS_OTHER:   the other compiler flags
#  LibDAI_VERSION:        version of the module
#  LibDAI_PREFIX:         prefix-directory of the module
#  LibDAI_INCLUDEDIR:     include-dir of the module
#  LibDAI_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(LibDAI libdai
  HEADERS dai/factorgraph.h dai/var.h
  LIBRARIES dai gmpxx gmp
  )
