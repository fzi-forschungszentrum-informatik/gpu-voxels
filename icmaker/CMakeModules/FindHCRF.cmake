# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find HCRF.  Once done, this will define:
#  HCRF_FOUND:          System has HCRF
#  HCRF_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  HCRF_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  HCRF_DEFINITIONS:    Preprocessor definitions.
#  HCRF_LIBRARIES:      only the libraries (w/o the '-l')
#  HCRF_LDFLAGS:        all required linker flags
#  HCRF_LDFLAGS_OTHER:  all other linker flags
#  HCRF_CFLAGS:         all required cflags
#  HCRF_CFLAGS_OTHER:   the other compiler flags
#  HCRF_VERSION:        version of the module
#  HCRF_PREFIX:         prefix-directory of the module
#  HCRF_INCLUDEDIR:     include-dir of the module
#  HCRF_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(HCRF hcrf
  HEADERS hCRF/hCRF.h
  LIBRARIES cgDescent hCRF lbfgs uncoptim
  )
