# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Ltdl.  Once done, this will define:
#  Ltdl_FOUND:          System has Ltdl
#  Ltdl_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Ltdl_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Ltdl_DEFINITIONS:    Preprocessor definitions.
#  Ltdl_LIBRARIES:      only the libraries (w/o the '-l')
#  Ltdl_LDFLAGS:        all required linker flags
#  Ltdl_LDFLAGS_OTHER:  all other linker flags
#  Ltdl_CFLAGS:         all required cflags
#  Ltdl_CFLAGS_OTHER:   the other compiler flags
#  Ltdl_VERSION:        version of the module
#  Ltdl_PREFIX:         prefix-directory of the module
#  Ltdl_INCLUDEDIR:     include-dir of the module
#  Ltdl_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Ltdl ltdl
  HEADERS ltdl.h
  LIBRARIES ltdl
  )
