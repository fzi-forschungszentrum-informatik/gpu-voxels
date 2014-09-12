# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find CarMaker.  Once done, this will define:
#  CarMaker_FOUND:          System has CarMaker
#  CarMaker_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  CarMaker_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  CarMaker_DEFINITIONS:    Preprocessor definitions.
#  CarMaker_LIBRARIES:      only the libraries (w/o the '-l')
#  CarMaker_LDFLAGS:        all required linker flags
#  CarMaker_LDFLAGS_OTHER:  all other linker flags
#  CarMaker_CFLAGS:         all required cflags
#  CarMaker_CFLAGS_OTHER:   the other compiler flags
#  CarMaker_VERSION:        version of the module
#  CarMaker_PREFIX:         prefix-directory of the module
#  CarMaker_INCLUDEDIR:     include-dir of the module
#  CarMaker_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

find_package(LibUSB)

if (LibUSB_FOUND)
  libfind_lib_with_pkg_config(CarMaker carmaker
    HEADERS CarMaker.h
    LIBRARIES carmaker car ipgdriver ipgroad tametire
    EXECUTABLES CreateCarMakerAppInfo
    HINTS /opt/ipg/hil/linux
    DEFINE _IC_BUILDER_CARMAKER_
    )
endif ()

