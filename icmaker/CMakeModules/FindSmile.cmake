# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Smile.  Once done, this will define:
#  Smile_FOUND:          System has Smile
#  Smile_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Smile_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Smile_DEFINITIONS:    Preprocessor definitions.
#  Smile_LIBRARIES:      only the libraries (w/o the '-l')
#  Smile_LDFLAGS:        all required linker flags
#  Smile_LDFLAGS_OTHER:  all other linker flags
#  Smile_CFLAGS:         all required cflags
#  Smile_CFLAGS_OTHER:   the other compiler flags
#  Smile_VERSION:        version of the module
#  Smile_PREFIX:         prefix-directory of the module
#  Smile_INCLUDEDIR:     include-dir of the module
#  Smile_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Smile libsmile
  HEADERS smile.h
  LIBRARIES smile
  HINTS /opt/tools/smile
  HEADER_PATHS /opt/tools/smile
  LIBRARY_PATHS /opt/tools/smile
  DEFINE _IC_BUILDER_SMILE_
  )
