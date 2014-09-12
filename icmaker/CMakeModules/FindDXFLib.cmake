# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find DXFLib.  Once done, this will define:
#  DXFLib_FOUND:          System has DXFLib
#  DXFLib_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  DXFLib_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  DXFLib_DEFINITIONS:    Preprocessor definitions.
#  DXFLib_LIBRARIES:      only the libraries (w/o the '-l')
#  DXFLib_LDFLAGS:        all required linker flags
#  DXFLib_LDFLAGS_OTHER:  all other linker flags
#  DXFLib_CFLAGS:         all required cflags
#  DXFLib_CFLAGS_OTHER:   the other compiler flags
#  DXFLib_VERSION:        version of the module
#  DXFLib_PREFIX:         prefix-directory of the module
#  DXFLib_INCLUDEDIR:     include-dir of the module
#  DXFLib_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(DXFLib dxflib
  HEADERS dl_dxf.h
  LIBRARIES dxflib
  )
