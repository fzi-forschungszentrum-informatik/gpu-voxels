# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Iconv.  Once done, this will define:
#  Iconv_FOUND:          System has Iconv
#  Iconv_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Iconv_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Iconv_DEFINITIONS:    Preprocessor definitions.
#  Iconv_LIBRARIES:      only the libraries (w/o the '-l')
#  Iconv_LDFLAGS:        all required linker flags
#  Iconv_LDFLAGS_OTHER:  all other linker flags
#  Iconv_CFLAGS:         all required cflags
#  Iconv_CFLAGS_OTHER:   the other compiler flags
#  Iconv_VERSION:        version of the module
#  Iconv_PREFIX:         prefix-directory of the module
#  Iconv_INCLUDEDIR:     include-dir of the module
#  Iconv_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Iconv iconv
  HEADERS iconv.h
  LIBRARIES iconv
  )
