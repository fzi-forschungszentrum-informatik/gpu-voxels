# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find HMMlib.  Once done, this will define:
#  HMMlib_FOUND:          System has HMMlib
#  HMMlib_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  HMMlib_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  HMMlib_DEFINITIONS:    Preprocessor definitions.
#  HMMlib_LIBRARIES:      only the libraries (w/o the '-l')
#  HMMlib_LDFLAGS:        all required linker flags
#  HMMlib_LDFLAGS_OTHER:  all other linker flags
#  HMMlib_CFLAGS:         all required cflags
#  HMMlib_CFLAGS_OTHER:   the other compiler flags
#  HMMlib_VERSION:        version of the module
#  HMMlib_PREFIX:         prefix-directory of the module
#  HMMlib_INCLUDEDIR:     include-dir of the module
#  HMMlib_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(HMMlib hmmlib
  HEADERS HMMlib/hmm.hpp
  )

if (HMMlib_FOUND)
  set(HMMlib_CFLAGS -msse4 CACHE INTERNAL "")
endif()
