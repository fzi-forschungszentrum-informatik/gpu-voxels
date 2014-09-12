# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find OMPL.  Once done, this will define:
#  OMPL_FOUND:          System has OMPL
#  OMPL_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  OMPL_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  OMPL_DEFINITIONS:    Preprocessor definitions.
#  OMPL_LIBRARIES:      only the libraries (w/o the '-l')
#  OMPL_LDFLAGS:        all required linker flags
#  OMPL_LDFLAGS_OTHER:  all other linker flags
#  OMPL_CFLAGS:         all required cflags
#  OMPL_CFLAGS_OTHER:   the other compiler flags
#  OMPL_VERSION:        version of the module
#  OMPL_PREFIX:         prefix-directory of the module
#  OMPL_INCLUDEDIR:     include-dir of the module
#  OMPL_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(OMPL ompl
  HEADERS ompl/base/StateSampler.h
  LIBRARIES ompl
  HINTS /opt/local
  )
