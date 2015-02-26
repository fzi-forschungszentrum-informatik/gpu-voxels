# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Glib.  Once done, this will define:
#  Glib_FOUND:          System has Glib
#  Glib_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Glib_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Glib_DEFINITIONS:    Preprocessor definitions.
#  Glib_LIBRARIES:      only the libraries (w/o the '-l')
#  Glib_LDFLAGS:        all required linker flags
#  Glib_LDFLAGS_OTHER:  all other linker flags
#  Glib_CFLAGS:         all required cflags
#  Glib_CFLAGS_OTHER:   the other compiler flags
#  Glib_VERSION:        version of the module
#  Glib_PREFIX:         prefix-directory of the module
#  Glib_INCLUDEDIR:     include-dir of the module
#  Glib_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Glib glib-2.0
  HEADERS glib.h glibconfig.h
  LIBRARIES glib-2.0
  )
