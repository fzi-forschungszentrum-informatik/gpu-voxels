# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find OsmPBF.  Once done, this will define:
#  OsmPBF_FOUND:          System has OsmPBF
#  OsmPBF_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  OsmPBF_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  OsmPBF_DEFINITIONS:    Preprocessor definitions.
#  OsmPBF_LIBRARIES:      only the libraries (w/o the '-l')
#  OsmPBF_LDFLAGS:        all required linker flags
#  OsmPBF_LDFLAGS_OTHER:  all other linker flags
#  OsmPBF_CFLAGS:         all required cflags
#  OsmPBF_CFLAGS_OTHER:   the other compiler flags
#  OsmPBF_VERSION:        version of the module
#  OsmPBF_PREFIX:         prefix-directory of the module
#  OsmPBF_INCLUDEDIR:     include-dir of the module
#  OsmPBF_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(OsmPBF libosmpbf
  HEADERS osmpbf/osmpbf.h
  LIBRARIES libosmpbf.a
  )
