# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Pthread.  Once done, this will define:
#  Pthread_FOUND:          System has Pthread
#  Pthread_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Pthread_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Pthread_DEFINITIONS:    Preprocessor definitions.
#  Pthread_LIBRARIES:      only the libraries (w/o the '-l')
#  Pthread_LDFLAGS:        all required linker flags
#  Pthread_LDFLAGS_OTHER:  all other linker flags
#  Pthread_CFLAGS:         all required cflags
#  Pthread_CFLAGS_OTHER:   the other compiler flags
#  Pthread_VERSION:        version of the module
#  Pthread_PREFIX:         prefix-directory of the module
#  Pthread_INCLUDEDIR:     include-dir of the module
#  Pthread_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Pthread libpthread
  HEADERS pthread.h
  LIBRARIES pthread
  )
