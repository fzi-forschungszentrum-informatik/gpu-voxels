# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find GFLIP.  Once done, this will define:
#  GFLIP_FOUND:          System has GFLIP
#  GFLIP_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  GFLIP_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  GFLIP_DEFINITIONS:    Preprocessor definitions.
#  GFLIP_LIBRARIES:      only the libraries (w/o the '-l')
#  GFLIP_LDFLAGS:        all required linker flags
#  GFLIP_LDFLAGS_OTHER:  all other linker flags
#  GFLIP_CFLAGS:         all required cflags
#  GFLIP_CFLAGS_OTHER:   the other compiler flags
#  GFLIP_VERSION:        version of the module
#  GFLIP_PREFIX:         prefix-directory of the module
#  GFLIP_INCLUDEDIR:     include-dir of the module
#  GFLIP_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(GFLIP gflip
  HEADERS gflip/gflip_engine.hpp
  LIBRARIES gflip gflip_vocabulary
  HINTS /opt/local
  )
