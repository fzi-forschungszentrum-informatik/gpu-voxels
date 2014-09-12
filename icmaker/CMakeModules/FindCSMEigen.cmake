# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find CSMEigen.  Once done, this will define:
#  CSMEigen_FOUND:          System has CSMEigen
#  CSMEigen_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  CSMEigen_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  CSMEigen_DEFINITIONS:    Preprocessor definitions.
#  CSMEigen_LIBRARIES:      only the libraries (w/o the '-l')
#  CSMEigen_LDFLAGS:        all required linker flags
#  CSMEigen_LDFLAGS_OTHER:  all other linker flags
#  CSMEigen_CFLAGS:         all required cflags
#  CSMEigen_CFLAGS_OTHER:   the other compiler flags
#  CSMEigen_VERSION:        version of the module
#  CSMEigen_PREFIX:         prefix-directory of the module
#  CSMEigen_INCLUDEDIR:     include-dir of the module
#  CSMEigen_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(CSMEigen csm
  HEADERS csm/csm.h gsl_eigen/gsl_eigen.h
  LIBRARIES csm_eigen
  HINTS /opt/local
  )
