# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Florian Kuhnt <kuhnt@fzi.de>
# \date    2016-03-15
#
# Try to find redisclient.  Once done, this will define:
#  redisclient_FOUND:          System has redisclient
#  redisclient_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  redisclient_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  redisclient_DEFINITIONS:    Preprocessor definitions.
#  redisclient_LIBRARIES:      only the libraries (w/o the '-l')
#  redisclient_LDFLAGS:        all required linker flags
#  redisclient_LDFLAGS_OTHER:  all other linker flags
#  redisclient_CFLAGS:         all required cflags
#  redisclient_CFLAGS_OTHER:   the other compiler flags
#  redisclient_VERSION:        version of the module
#  redisclient_PREFIX:         prefix-directory of the module
#  redisclient_INCLUDEDIR:     include-dir of the module
#  redisclient_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(redisclient redisclient
  HINTS /usr/local/include/src
  HEADERS redisclient/redisasyncclient.h
  )
