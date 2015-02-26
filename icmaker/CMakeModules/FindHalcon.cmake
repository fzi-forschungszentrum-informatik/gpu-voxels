# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Steffen Rühl <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find Halcon.  Once done, this will define:
#  Halcon_FOUND:          System has Halcon
#  Halcon_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Halcon_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Halcon_DEFINITIONS:    Preprocessor definitions.
#  Halcon_LIBRARIES:      only the libraries (w/o the '-l')
#  Halcon_LDFLAGS:        all required linker flags
#  Halcon_LDFLAGS_OTHER:  all other linker flags
#  Halcon_CFLAGS:         all required cflags
#  Halcon_CFLAGS_OTHER:   the other compiler flags
#  Halcon_VERSION:        version of the module
#  Halcon_PREFIX:         prefix-directory of the module
#  Halcon_INCLUDEDIR:     include-dir of the module
#  Halcon_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------


include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Halcon halcon
  HEADER_PATHS "/opt/halcon/include"
  #HEADER_PATHS "/opt/halcon11/include"
  HEADERS HalconC.h
  LIBRARY_PATHS "/opt/halcon/lib/x86sse2-linux2.4-gcc40"
  #LIBRARY_PATHS "/opt/halcon11/lib/x64-linux2.4-gcc40"
  LIBRARIES halconcpp halcon
  
  )

# Add a subdirectory needed by halcon.
if (Halcon_FOUND)
  LIST(GET Halcon_INCLUDE_DIRS 0 dir)
  SET(subdir "${dir}/halconcpp")
  LIST(APPEND Halcon_INCLUDE_DIRS ${subdir})
endif ()
