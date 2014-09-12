# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Ghmm.  Once done, this will define:
#  Ghmm_FOUND:          System has Ghmm
#  Ghmm_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Ghmm_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Ghmm_DEFINITIONS:    Preprocessor definitions.
#  Ghmm_LIBRARIES:      only the libraries (w/o the '-l')
#  Ghmm_LDFLAGS:        all required linker flags
#  Ghmm_LDFLAGS_OTHER:  all other linker flags
#  Ghmm_CFLAGS:         all required cflags
#  Ghmm_CFLAGS_OTHER:   the other compiler flags
#  Ghmm_VERSION:        version of the module
#  Ghmm_PREFIX:         prefix-directory of the module
#  Ghmm_INCLUDEDIR:     include-dir of the module
#  Ghmm_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Ghmm ghmm
  HEADERS ghmm/ghmm.h
  LIBRARIES ghmm
  )
