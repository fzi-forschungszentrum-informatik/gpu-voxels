# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find CarMakerRealTime.  Once done, this will define:
#  CarMakerRealTime_FOUND:          System has CarMakerRealTime
#  CarMakerRealTime_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  CarMakerRealTime_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  CarMakerRealTime_DEFINITIONS:    Preprocessor definitions.
#  CarMakerRealTime_LIBRARIES:      only the libraries (w/o the '-l')
#  CarMakerRealTime_LDFLAGS:        all required linker flags
#  CarMakerRealTime_LDFLAGS_OTHER:  all other linker flags
#  CarMakerRealTime_CFLAGS:         all required cflags
#  CarMakerRealTime_CFLAGS_OTHER:   the other compiler flags
#  CarMakerRealTime_VERSION:        version of the module
#  CarMakerRealTime_PREFIX:         prefix-directory of the module
#  CarMakerRealTime_INCLUDEDIR:     include-dir of the module
#  CarMakerRealTime_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

find_package(LibUSB REQUIRED)

libfind_lib_with_pkg_config(CarMakerRealTime carmakerrealtime
  HEADERS CarMaker.h
  LIBRARIES carmaker car ipgdriver ipgroad usb native rtdm SensoDrive xenomai tametire
  EXECUTABLES CreateCarMakerAppInfo
  HINTS /opt/ipg/hil/linux-xeno
  DEFINE _IC_BUILDER_CARMAKER_
  )
