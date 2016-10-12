# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------

#----------------------------------------------------------------------
# \file
#
# \author  Florian Kuhnt <kuhnt@fzi.de>
# \date    2016-07-30
#
# Try to find Pugixml.  Once done, this will define:
#  Pugixml_FOUND:          System has Pugixml
#  Pugixml_INCLUDE_DIRS:   The '-I' preprocessor flags (w/o the '-I')
#  Pugixml_LIBRARY_DIRS:   The paths of the libraries (w/o the '-L')
# Variables defined if pkg-config was employed:
#  Pugixml_DEFINITIONS:    Preprocessor definitions.
#  Pugixml_LIBRARIES:      only the libraries (w/o the '-l')
#  Pugixml_LDFLAGS:        all required linker flags
#  Pugixml_LDFLAGS_OTHER:  all other linker flags
#  Pugixml_CFLAGS:         all required cflags
#  Pugixml_CFLAGS_OTHER:   the other compiler flags
#  Pugixml_VERSION:        version of the module
#  Pugixml_PREFIX:         prefix-directory of the module
#  Pugixml_INCLUDEDIR:     include-dir of the module
#  Pugixml_LIBDIR:         lib-dir of the module
#----------------------------------------------------------------------

include(PrintLibraryStatus)
include(LibFindMacros)

libfind_lib_with_pkg_config(Pugixml pugixml
  HEADERS pugixml.hpp
  LIBRARIES pugixml
  )
