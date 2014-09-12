# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Jan Oberlaender <oberlaender@fzi.de>
# \date    2014-08-13
#
# Try to find PCAP.  Once done, this will define:
#  PCAP_FOUND:          System has PCAP
#  PCAP_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  PCAP_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  PCAP_DEFINITIONS:    Preprocessor definitions.
#  PCAP_LIBRARIES:      only the libraries (w/o the '-l')
#  PCAP_LDFLAGS:        all required linker flags
#  PCAP_LDFLAGS_OTHER:  all other linker flags
#  PCAP_CFLAGS:         all required cflags
#  PCAP_CFLAGS_OTHER:   the other compiler flags
#  PCAP_VERSION:        version of the module
#  PCAP_PREFIX:         prefix-directory of the module
#  PCAP_INCLUDEDIR:     include-dir of the module
#  PCAP_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(PCAP pcap
  HEADERS pcap/pcap.h
  LIBRARIES pcap
  )
