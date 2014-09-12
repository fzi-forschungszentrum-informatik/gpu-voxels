# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find NetSNMP.  Once done, this will define:
#  NetSNMP_FOUND:          System has NetSNMP
#  NetSNMP_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  NetSNMP_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  NetSNMP_DEFINITIONS:    Preprocessor definitions.
#  NetSNMP_LIBRARIES:      only the libraries (w/o the '-l')
#  NetSNMP_LDFLAGS:        all required linker flags
#  NetSNMP_LDFLAGS_OTHER:  all other linker flags
#  NetSNMP_CFLAGS:         all required cflags
#  NetSNMP_CFLAGS_OTHER:   the other compiler flags
#  NetSNMP_VERSION:        version of the module
#  NetSNMP_PREFIX:         prefix-directory of the module
#  NetSNMP_INCLUDEDIR:     include-dir of the module
#  NetSNMP_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(NetSNMP netsnmp
  HEADERS net-snmp/net-snmp-config.h
  LIBRARIES netsnmp
  )
