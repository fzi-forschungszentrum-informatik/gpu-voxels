# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find HbFoundationExtract.  Once done, this will define:
#  HbFoundationExtract_FOUND:          System has HbFoundationExtract
#  HbFoundationExtract_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  HbFoundationExtract_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  HbFoundationExtract_DEFINITIONS:    Preprocessor definitions.
#  HbFoundationExtract_LIBRARIES:      only the libraries (w/o the '-l')
#  HbFoundationExtract_LDFLAGS:        all required linker flags
#  HbFoundationExtract_LDFLAGS_OTHER:  all other linker flags
#  HbFoundationExtract_CFLAGS:         all required cflags
#  HbFoundationExtract_CFLAGS_OTHER:   the other compiler flags
#  HbFoundationExtract_VERSION:        version of the module
#  HbFoundationExtract_PREFIX:         prefix-directory of the module
#  HbFoundationExtract_INCLUDEDIR:     include-dir of the module
#  HbFoundationExtract_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(HbFoundationExtract HbFoundationExtract
  HEADERS HBSTypes.hpp
  LIBRARIES hb_foundation_extract
  )
