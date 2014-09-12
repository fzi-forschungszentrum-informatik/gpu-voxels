# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find CSM.  Once done, this will define:
#  CSM_FOUND:          System has CSM
#  CSM_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  CSM_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  CSM_DEFINITIONS:    Preprocessor definitions.
#  CSM_LIBRARIES:      only the libraries (w/o the '-l')
#  CSM_LDFLAGS:        all required linker flags
#  CSM_LDFLAGS_OTHER:  all other linker flags
#  CSM_CFLAGS:         all required cflags
#  CSM_CFLAGS_OTHER:   the other compiler flags
#  CSM_VERSION:        version of the module
#  CSM_PREFIX:         prefix-directory of the module
#  CSM_INCLUDEDIR:     include-dir of the module
#  CSM_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(CSM csm
  HEADERS csm/csm.h
  LIBRARIES csm
  )
