# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Sebastian Klemm <klemm@fzi.de>
# \date    2014-10-29
#
# Try to find NLopt (http://ab-initio.mit.edu/wiki/index.php/NLopt)
# Once done, this will define:
#  NLopt_FOUND:          System has NLopt
#  NLopt_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  NLopt_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  NLopt_DEFINITIONS:    Preprocessor definitions.
#  NLopt_LIBRARIES:      only the libraries (w/o the '-l')
#  NLopt_LDFLAGS:        all required linker flags
#  NLopt_LDFLAGS_OTHER:  all other linker flags
#  NLopt_CFLAGS:         all required cflags
#  NLopt_CFLAGS_OTHER:   the other compiler flags
#  NLopt_VERSION:        version of the module
#  NLopt_PREFIX:         prefix-directory of the module
#  NLopt_INCLUDEDIR:     include-dir of the module
#  NLopt_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(NLopt nlopt
  HEADERS nlopt.h
  HEADER_PATHS "/usr/include"
  DEFINE _IC_BUILDER_NLOPT_
  )
