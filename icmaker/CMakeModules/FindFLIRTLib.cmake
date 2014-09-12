# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find FLIRTLib.  Once done, this will define:
#  FLIRTLib_FOUND:          System has FLIRTLib
#  FLIRTLib_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  FLIRTLib_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  FLIRTLib_DEFINITIONS:    Preprocessor definitions.
#  FLIRTLib_LIBRARIES:      only the libraries (w/o the '-l')
#  FLIRTLib_LDFLAGS:        all required linker flags
#  FLIRTLib_LDFLAGS_OTHER:  all other linker flags
#  FLIRTLib_CFLAGS:         all required cflags
#  FLIRTLib_CFLAGS_OTHER:   the other compiler flags
#  FLIRTLib_VERSION:        version of the module
#  FLIRTLib_PREFIX:         prefix-directory of the module
#  FLIRTLib_INCLUDEDIR:     include-dir of the module
#  FLIRTLib_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(FLIRTLib flirtlib
  HEADERS flirtlib/feature/InterestPoint.h
  LIBRARIES flirt_sensors flirt_sensorstream flirt_geometry flirt_feature flirt_utils
  HINTS /opt/local
  )
