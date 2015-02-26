# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Klaus Fischnaller <fischnal@fzi.de>
# \date    2015-01-23
#
# Try to find ZMQ.  Once done, this will define:
#  ZMQ_FOUND:          System has Reflexx
#  ZMQ_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  ZMQ_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  ZMQ_DEFINITIONS:    Preprocessor definitions.
#  ZMQ_LIBRARIES:      only the libraries (w/o the '-l')
#  ZMQ_LDFLAGS:        all required linker flags
#  ZMQ_LDFLAGS_OTHER:  all other linker flags
#  ZMQ_CFLAGS:         all required cflags
#  ZMQ_CFLAGS_OTHER:   the other compiler flags
#  ZMQ_VERSION:        version of the module
#  ZMQ_PREFIX:         prefix-directory of the module
#  ZMQ_INCLUDEDIR:     include-dir of the module
#  ZMQ_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(ZMQ zmq
  HEADERS zmq.hpp
  LIBRARIES zmq pthread
  )
