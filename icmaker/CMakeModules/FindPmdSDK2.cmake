# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find PmdSDK2.  Once done, this will define:
#  PmdSDK2_FOUND:          System has PmdSDK2
#  PmdSDK2_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  PmdSDK2_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  PmdSDK2_DEFINITIONS:    Preprocessor definitions.
#  PmdSDK2_LIBRARIES:      only the libraries (w/o the '-l')
#  PmdSDK2_LDFLAGS:        all required linker flags
#  PmdSDK2_LDFLAGS_OTHER:  all other linker flags
#  PmdSDK2_CFLAGS:         all required cflags
#  PmdSDK2_CFLAGS_OTHER:   the other compiler flags
#  PmdSDK2_VERSION:        version of the module
#  PmdSDK2_PREFIX:         prefix-directory of the module
#  PmdSDK2_INCLUDEDIR:     include-dir of the module
#  PmdSDK2_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(PmdSDK2 libPmdSDK2
  HEADERS pmdsdk2.h
  LIBRARIES pmdaccess2
  DEFINE _IC_BUILDER_PMDSDK2_
  )
