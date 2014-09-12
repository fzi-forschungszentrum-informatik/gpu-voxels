# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find SQLite3.  Once done, this will define:
#  SQLite3_FOUND:          System has SQLite3
#  SQLite3_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  SQLite3_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  SQLite3_DEFINITIONS:    Preprocessor definitions.
#  SQLite3_LIBRARIES:      only the libraries (w/o the '-l')
#  SQLite3_LDFLAGS:        all required linker flags
#  SQLite3_LDFLAGS_OTHER:  all other linker flags
#  SQLite3_CFLAGS:         all required cflags
#  SQLite3_CFLAGS_OTHER:   the other compiler flags
#  SQLite3_VERSION:        version of the module
#  SQLite3_PREFIX:         prefix-directory of the module
#  SQLite3_INCLUDEDIR:     include-dir of the module
#  SQLite3_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(SQLite3 sqlite3
  HEADERS sqlite3.h
  LIBRARIES sqlite3
  HINTS /opt/local
  )
