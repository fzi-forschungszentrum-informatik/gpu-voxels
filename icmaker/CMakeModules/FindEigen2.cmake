# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Eigen2.  Once done, this will define:
#  Eigen2_FOUND:          System has Eigen2
#  Eigen2_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Eigen2_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Eigen2_DEFINITIONS:    Preprocessor definitions.
#  Eigen2_LIBRARIES:      only the libraries (w/o the '-l')
#  Eigen2_LDFLAGS:        all required linker flags
#  Eigen2_LDFLAGS_OTHER:  all other linker flags
#  Eigen2_CFLAGS:         all required cflags
#  Eigen2_CFLAGS_OTHER:   the other compiler flags
#  Eigen2_VERSION:        version of the module
#  Eigen2_PREFIX:         prefix-directory of the module
#  Eigen2_INCLUDEDIR:     include-dir of the module
#  Eigen2_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Eigen2 eigen2
  HEADERS Eigen/Core
  HEADER_PATHS "/usr/include/eigen2" "${CMAKE_INSTALL_PREFIX}/include/eigen2"
  DEFINE _IC_BUILDER_EIGEN_
  )
