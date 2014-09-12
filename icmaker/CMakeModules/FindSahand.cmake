# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Sahand.  Once done, this will define:
#  Sahand_FOUND:          System has Sahand
#  Sahand_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Sahand_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Sahand_DEFINITIONS:    Preprocessor definitions.
#  Sahand_LIBRARIES:      only the libraries (w/o the '-l')
#  Sahand_LDFLAGS:        all required linker flags
#  Sahand_LDFLAGS_OTHER:  all other linker flags
#  Sahand_CFLAGS:         all required cflags
#  Sahand_CFLAGS_OTHER:   the other compiler flags
#  Sahand_VERSION:        version of the module
#  Sahand_PREFIX:         prefix-directory of the module
#  Sahand_INCLUDEDIR:     include-dir of the module
#  Sahand_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Sahand sahand
  HEADERS sahand/SAHandCtrlApi.h
  LIBRARIES libsahand-static.a
  )
